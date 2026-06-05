from nndbg import ModelProbe, DeadNeuronDetector, set_verbose

set_verbose(False)
probe = ModelProbe.from_pretrained("distilbert-base-uncased")

probe.add_axis("language", {
    "english": [
        "The cat sat on the mat.",
        "Scientists discovered a new species in the Amazon.",
        "The stock market crashed today.",
    ],
    "french": [
        "Le chat était assis sur le tapis.",
        "Les scientifiques ont découvert une nouvelle espèce.",
        "Le marché boursier s'est effondré aujourd'hui.",
    ],
})

probe.add_axis("domain", {
    "code": [
        "def forward(self, x): return self.layer(x)",
        "optimizer.zero_grad()",
        "loss = criterion(output, target)",
    ],
    "medical": [
        "Patient presents with acute respiratory symptoms.",
        "Administer 500mg of amoxicillin twice daily.",
        "Blood pressure readings were consistently elevated.",
    ],
})

print("Running ModelProbe...")
results = probe.run()
print("Run complete.")

print("\nProbe scores:")
for axis, scores in results.probe_scores.items():
    print(f"  {axis}:")
    for layer, score in scores.items():
        print(f"    {layer}: {score:.4f}")

print("\nFitting DeadNeuronDetector for 'language' axis...")
detector = DeadNeuronDetector(probe)
detector.fit("language", raw_activations=results._raw_activations)
report = detector.report("language")

print("\nDead neuron report summary:")
print(report.to_dict())
print("\nRich report output:")
report.show(show_neurons=True)
