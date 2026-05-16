import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def plot_training_curve(values, title):

    plt.figure(figsize=(8, 4))

    plt.plot(values)

    plt.title(title)

    plt.xlabel("Epoch")

    plt.ylabel("Value")

    plt.grid(True)

    plt.show()


def plot_confusion_matrix(cm):

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.show()


def compare_models(names, scores, metric_name):

    plt.figure(figsize=(8, 4))

    bars = plt.bar(names, scores)

    plt.title(
        f"Comparison of {metric_name}"
    )

    plt.ylabel(metric_name)

    min_score = min(scores)
    max_score = max(scores)
    
    if "Tempo" not in metric_name:
        if max_score - min_score < 0.1:
            plt.ylim(max(0, min_score - 0.01), min(1.05, max_score + 0.01))
        else:
            plt.ylim(0, 1.05)
    else:
        plt.ylim(0, max_score * 1.1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.002 if "Tempo" not in metric_name else max_score * 0.02), f'{yval:.0f}' if "Tempo" in metric_name else f'{yval:.4f}', ha='center', va='bottom')

    plt.xticks(rotation=15)

    plt.show()
