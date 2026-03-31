import json

# Generate unique sentences to avoid tokenizer compression
lines = []
colors = ["red", "blue", "green", "yellow", "orange"]
for i in range(3000):
    color = colors[i % 5]
    val = (i * 17) % 997
    lines.append("Sentence number %d: The value of x at position %d is %d and the color is %s." % (i, i, val, color))

text = " ".join(lines)

# Generate prompts at different sizes
for count in [500, 1000, 1500, 2000, 2500, 3000]:
    subset = " ".join(lines[:count])
    payload = json.dumps({
        "model": "qwen3.5-4b",
        "messages": [{"role": "user", "content": subset + " What was the color mentioned in sentence number %d?" % (count - 1)}],
        "max_tokens": 64
    })
    with open("/tmp/stress_%d.json" % count, "w")  as f:
        f.write(payload)
    print("Generated %d sentences, payload %d bytes, ~%d words" % (count, len(payload), len(subset.split())))
