from django import forms

style = forms.Select(attrs={"class": "form-control"})


class NLPForm(forms.Form):
    prompt = forms.CharField(required=True, widget=forms.Textarea(attrs={"rows":"5"}))
