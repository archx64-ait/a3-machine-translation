import torchtext, pickle

torchtext.disable_torchtext_deprecation_warning()

data = pickle.load(open('models/transforms-additive.pkl'))


