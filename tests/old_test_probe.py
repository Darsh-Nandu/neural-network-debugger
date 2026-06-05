from nndbg import ModelProbe
from nndbg.probing.trainer import ProbeTrainer

trainer = ProbeTrainer(
    cv_folds=5,
    max_iter=2000,
    features=['mean','std'],
    test_size=0.2
)

probe = ModelProbe.from_pretrained(
    "bert-base-multilingual-cased",
    probe_trainer = trainer
    )

probe.add_axis("language", {
    "english": [
        "The cat sat on the mat.",
        "Scientists discovered a new species in the Amazon.",
        "The stock market crashed today.",
        "She opened the door slowly and looked inside.",
        "It was raining heavily outside the window.",
        "He decided to take a walk in the park.",
        "The government announced new economic policies.",
        "Children were playing in the street.",
        "The doctor recommended a new treatment.",
        "Technology is changing the world rapidly.",
    ],
    "french": [
        "Le chat était assis sur le tapis.",
        "Les scientifiques ont découvert une nouvelle espèce.",
        "Le marché boursier s'est effondré aujourd'hui.",
        "Elle a ouvert la porte lentement et a regardé à l'intérieur.",
        "Il pleuvait très fort dehors.",
        "Il a décidé de se promener dans le parc.",
        "Le gouvernement a annoncé de nouvelles politiques.",
        "Les enfants jouaient dans la rue.",
        "Le médecin a recommandé un nouveau traitement.",
        "La technologie change le monde rapidement.",
    ],
    "hindi": [
        "बिल्ली चटाई पर बैठी थी।",
        "वैज्ञानिकों ने अमेज़न में एक नई प्रजाति खोजी।",
        "आज शेयर बाजार गिर गया।",
        "उसने धीरे से दरवाजा खोला और अंदर देखा।",
        "खिड़की के बाहर जोरों से बारिश हो रही थी।",
        "उसने पार्क में टहलने का फैसला किया।",
        "सरकार ने नई आर्थिक नीतियों की घोषणा की।",
        "बच्चे सड़क पर खेल रहे थे।",
        "डॉक्टर ने एक नए उपचार की सिफारिश की।",
        "तकनीक तेजी से दुनिया बदल रही है।",
    ],
})

probe.add_axis("domain", {
    "legal": [
        "The defendant shall appear before the court.",
        "The contract is subject to the laws of the jurisdiction.",
        "The plaintiff filed a motion to dismiss the case.",
        "All parties must comply with the terms of the agreement.",
        "The judge ruled in favor of the appellant.",
        "Evidence submitted must be admissible under federal law.",
        "The attorney requested a continuance of the hearing.",
        "Liability shall be determined by the court of law.",
        "The arbitration clause is binding on both parties.",
        "The verdict was overturned on appeal.",
    ],
    "medical": [
        "Patient presents with acute respiratory symptoms.",
        "The MRI scan revealed a lesion in the frontal lobe.",
        "Administer 500mg of amoxicillin twice daily.",
        "The patient was diagnosed with type 2 diabetes.",
        "Blood pressure readings were consistently elevated.",
        "The surgeon performed a minimally invasive procedure.",
        "Post-operative care includes wound monitoring.",
        "The biopsy confirmed malignant tissue growth.",
        "Prescribed anticoagulants to prevent thrombosis.",
        "The patient reported chronic lower back pain.",
    ],
    "code": [
        "def forward(self, x): return self.layer(x)",
        "import torch.nn as nn",
        "for epoch in range(100): optimizer.step()",
        "model.eval() with torch.no_grad():",
        "loss = criterion(output, target)",
        "class TransformerBlock(nn.Module):",
        "x = F.relu(self.fc1(x))",
        "optimizer = torch.optim.Adam(model.parameters())",
        "batch_size = 32 learning_rate = 0.001",
        "tokenizer.encode(text, return_tensors='pt')",
    ],
})

results = probe.run()

print(results.summary(top_k=None))