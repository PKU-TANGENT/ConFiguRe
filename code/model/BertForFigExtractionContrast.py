from transformers.models.bert.modeling_bert import *


def batched_index_select(input, dim, index):
    """
    helper function for batched index select
    """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class BertForFigExtractionContrast(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.contrast_lambda = config.contrast_lambda

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.affine = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        sent_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        epoch_num = None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output_hidden_states = outputs[0]
        # output_punc_hid = batched_index_select(output_hidden_states, 1, punc_idx) # bsz * max_punc_num * hidden_size

        sent_embs_all = []

        bsz = input_ids.size()[0]
        max_sent_len = input_ids.size()[1]
        max_sent_id = torch.max(sent_ids)
        sent_ids_matrix = (torch.arange(max_sent_id) + 1).unsqueeze(0).unsqueeze(-1).expand(bsz, -1, max_sent_len).to(input_ids.device)
        sent_ids_matrix2 = sent_ids.unsqueeze(1).expand(-1, max_sent_id, -1)
        select_matrix = (sent_ids_matrix == sent_ids_matrix2).float()
        word_num = torch.sum(select_matrix, dim=-1).unsqueeze(-1).expand(-1, -1, max_sent_len)
        pooling_matrix = torch.where(word_num > 0, select_matrix / word_num, select_matrix)
        sent_embs = torch.bmm(pooling_matrix, output_hidden_states)

        input_embs = self.bert.embeddings.word_embeddings(input_ids) # bsz * seq_len * hid_size
        input_embs = torch.bmm(pooling_matrix, input_embs) # bsz * sent_num_all * hid_size
        input_embs = self.affine(input_embs)  # bsz * sent_num_all * hid_size

        Cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        sent_input_dist = torch.exp(1 - Cos(input_embs, sent_embs))
        mask_is_fig = labels >= 1
        
        denominator = torch.sum(sent_input_dist, dim=-1)
        numerator = torch.sum(sent_input_dist * mask_is_fig, dim=-1)
        contrast_loss = torch.sum(-torch.log(numerator/denominator))

        # sent_embs_all = torch.cat(sent_embs_all, dim=0)
        output_punc_hid = self.dropout(torch.concat((sent_embs, input_embs), dim=-1))
        logits = self.classifier(output_punc_hid) # bsz * sent_num_all * num_label

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1) # notice!
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + contrast_loss * self.contrast_lambda

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )