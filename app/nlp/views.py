from django.views.generic import TemplateView
from django.views.generic.edit import FormView
from nlp.forms import NLPForm
from typing import Any
from nlp.utils_scratch import *
import torch


class IndexView(TemplateView):
    template_name = "index.html"


class SuccessView(TemplateView):
    template_name = "success.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        result = self.request.GET.get("result")

        try:
            # Add the result to the context
            context["result"] = result

        except ValueError:
            context["result"] = [""]

        return context


class NLPFormView(FormView):
    form_class = NLPForm
    template_name = "nlp.html"
    seq_length = 128
    model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    def translate(self, prompt):
        # transform the input text into source tokens
        source_tokens = text_transform[SRC_LANGUAGE](prompt).to(device)
        source_tokens = source_tokens.reshape(1, -1)

        # initialize the input tokens with the end-of-sequence token
        input_token_sequence = [EOS_IDX]
        output_token_sequence = []

        # create the source mask for the encoder
        source_mask = self.model.make_src_mask(source_tokens)

        # encode the source tokens
        with torch.no_grad():
            encoded_output = self.model.encoder(source_tokens, source_mask)

        # generate the target tokens sequentially
        for i in range(self.seq_length):
            with torch.no_grad():

                # prepare the input tokens for the decoder
                decoder_input = torch.LongTensor(input_token_sequence).unsqueeze(0).to(device)
                target_mask = self.model.make_trg_mask(decoder_input)

                # eecode the output
                decoder_output, _ = self.model.decoder(
                    decoder_input, encoded_output, target_mask, source_mask
                )
                predicted_token = decoder_output.argmax(2)[:, -1].item()

            # add the predicted token to the input and output sequences
            input_token_sequence.append(predicted_token)
            output_token_sequence.append(predicted_token)

            # stop the loop as soon as the end-of-sequence token is predicted
            if predicted_token == EOS_IDX:
                break

        # change the output tokens to the target language text
        target_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[token] for token in output_token_sequence]
        translated_text = "".join(target_tokens[1:-1])

        return translated_text

    def form_valid(self, form):
        prompt = form.cleaned_data["prompt"]
        result = self.translate(prompt=prompt)
        context = self.get_context_data(result=result)
        print(context)
        return self.render_to_response(context)

    def form_invalid(self, form):
        return super().form_invalid(form)

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        # context["results"] = getattr(self, "result", None)
        context["result"] = kwargs.get("result", None)
        return context
