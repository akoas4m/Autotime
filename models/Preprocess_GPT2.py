import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}' if torch.cuda.is_available() and configs.gpu >= 0 else 'cpu')
        print(self.device)

        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.vocab_size = self.gpt2_tokenizer.vocab_size
        self.hidden_dim_of_gpt2 = 768  # GPT-2 base model hidden dimension

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.gpt2_tokenizer(x, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.device)
        result = self.gpt2.transformer.wte(output)  # GPT-2's word token embeddings
        return result

    def forecast(self, x_mark_enc):
        # x_mark_enc: [bs x T x hidden_dim_of_gpt2]
        x_mark_enc = torch.cat([self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))], 0)
        text_outputs = self.gpt2.transformer(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs

    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)