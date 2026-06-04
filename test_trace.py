from nndbg import ModelProbe

probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

# Single language trace
french = probe.trace("Le chat était assis sur le tapis.")
print(french.summary())
french.show()

# Find key layers
print(french.most_active(top_k=5))
print(french.stable_at())

# Raw tensor access
tensor = french.activations["encoder.layer.4"]
print(tensor.shape)

# Compare two languages
english = probe.trace("The cat sat on the mat.")
english.compare(french)