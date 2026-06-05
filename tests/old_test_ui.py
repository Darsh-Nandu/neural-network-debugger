from nndbg import ModelProbe

probe = ModelProbe.from_pretrained("bert-base-multilingual-cased")

# Trace
french = probe.trace("Le chat était assis sur le tapis.")
french.summary()

# Full run
probe.add_axis("language", {
    "english": ["The cat sat.", "Hello world."] * 5,
    "french":  ["Le chat.", "Bonjour monde."] * 5,
})
results = probe.run()
results.summary()