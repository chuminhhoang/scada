from scada_classifier import ScadaClassifier
import config
cfg = config.CLASSIFY.clone()
scadaclassifier = ScadaClassifier(cfg)
# x1, y1, x2, y2
pred = scadaclassifier.classify('data_test/U_o6HQ_x.jpg', [1003.3296, 408.51, 1135.36992, 628.59024])
print(pred)