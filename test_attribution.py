from nndbg import ModelProbe
from nndbg.attribution import NeuronAttributor, HeadAttributor, InterferenceDetector
from nndbg.probing.trainer import ProbeTrainer

probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

probe.add_axis("language", {
    "english": ["The cat sat on the mat."] * 5,
    "french":  ["Le chat était assis."] * 5,
    "hindi":   ["बिल्ली चटाई पर बैठी थी।"] * 5,
})

probe.add_axis("domain", {
    "legal":   ["The defendant shall appear before court."] * 5,
    "medical": ["Patient presents with acute symptoms."] * 5,
})

# -- Neuron Attribution --
attr = NeuronAttributor(probe)
attr.fit("language")
attr.fit("domain")

attr.show("language", group="french", top_k=10)
attr.show("language", group="french", top_k=10, mode="exclusive")

# -- Head Attribution --
heads = HeadAttributor(probe)
heads.fit("language")
heads.show("language", top_k=10)
heads.plot_heatmap("language")

# -- Interference --
detector = InterferenceDetector(attr)
detector.check("language", "domain")