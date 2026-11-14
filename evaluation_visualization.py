
import matplotlib.pyplot as plt

# Replace these example values with
# your actual evaluation output.

metrics = {
    "Precision@5": 0.60,
    "Recall@5": 0.75,
    "F1@5": 0.67,
    "MRR": 0.80
}

names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(9, 5))

plt.bar(names, values)

plt.title("AI Job Recommendation Model Performance")

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")

plt.ylim(0, 1)

for i, value in enumerate(values):

    plt.text(
        i,
        value + 0.02,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "model_performance.png",
    dpi=300
)

plt.show()

print("Performance chart saved successfully!")
