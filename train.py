from cv2 import log
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.datasets

from normalizing_flow import NormalizingFlow
from torch.utils.tensorboard import SummaryWriter


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_writer = SummaryWriter(log_dir="logs/exp2")
    save_per_iter = 25

    use_trained_model = False
    traind_model_path = "trained_model_exp2.pth"
    save_model_path = "trained_model.pth"

    target_kind = "double_moon"

    num_iter = 300
    batch_size = 8192
    learning_rate = 1e-3
    feature_dim = 2
    hidden_dim = 128
    num_layers = 10


def run_sampling(num_samples: int, kind: str = Config.target_kind):
    if kind == "eclipse":
        return samples_from_eclipse(num_samples)
    elif kind == "double_moon":
        return samples_from_double_moon(num_samples)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def samples_from_eclipse(
    num_samples: int, a: float = 2.0, b: float = 1.0, epislon: float = 0.1
):
    theta = 2 * np.pi * torch.rand(num_samples)
    x = a * torch.cos(theta)
    y = b * torch.sin(theta)

    points = torch.stack([x, y], dim=1)
    noise = epislon * torch.randn(num_samples, 2)

    assert points.shape[0] == num_samples
    assert points.shape[1] == 2

    return points + noise


def samples_from_double_moon(num_samples: int, d: float = 1.0):
    import random

    data = sklearn.datasets.make_moons(
        n_samples=random.randint(num_samples, 2 * num_samples + 1000), noise=0.1
    )
    x = torch.tensor(data[0][:num_samples, 0], dtype=torch.float32)
    y = torch.tensor(data[0][:num_samples, 1], dtype=torch.float32)
    points = torch.stack([x, y], dim=1)

    return points


def train(flow_model: NormalizingFlow):
    summary_writer = Config.summary_writer

    optimizer = torch.optim.Adam(flow_model.parameters(), lr=Config.learning_rate)

    for i in range(Config.num_iter):
        flow_model.train()
        x = run_sampling(Config.batch_size).to(Config.device)
        log_likelihood = flow_model.log_likelihood(x)
        loss = -log_likelihood.mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        summary_writer.add_scalar("loss", loss.item(), i)

        print(f"Iter: {i}, Loss: {loss.item()}")

        if i % Config.save_per_iter == 0:
            flow_model.eval()
            z = flow_model.sample_from_prior(Config.batch_size).to(Config.device)
            x = flow_model.inverse(z).detach().cpu().numpy()

            # Plot the x on the 2D plane
            plt.figure(figsize=(6, 6))
            plt.scatter(x[:, 0], x[:, 1], s=1, c="blue", alpha=0.5)
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.title(f"Learned Distribution at iteration {i}")

            # Save the plot
            summary_writer.add_figure("learned_distribution", plt.gcf(), i)
            print(f"Saved plot at iteration {i}")

        summary_writer.flush()

    torch.save(flow_model.state_dict(), Config.save_model_path)

    return


if __name__ == "__main__":
    sns.set()

    flow_model = NormalizingFlow(
        feature_dim=Config.feature_dim,
        hidden_dim=Config.hidden_dim,
        num_layers=Config.num_layers,
    ).to(Config.device)

    prior_samples = flow_model.sample_from_prior(4096)
    # visualize the prior distribution
    plt.figure(figsize=(6, 6))
    plt.scatter(prior_samples[:, 0], prior_samples[:, 1], s=10, c="blue", alpha=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Prior Distribution")
    Config.summary_writer.add_figure("prior_distribution", plt.gcf())
    plt.close()

    if Config.use_trained_model:
        print("Loading trained model")
        flow_model.load_state_dict(torch.load(Config.traind_model_path))
    else:
        print("Training the model")
        train(flow_model)
        print("Model trained and saved")

    flow_model.eval()

    # visualize the learned distribution
    print("Visualizing the learned distribution")
    z = flow_model.sample_from_prior(4096).to(Config.device)

    summary_writer = Config.summary_writer

    for i in range(flow_model.num_layers + 1):
        print(f"calculating before {i} layers")
        x = flow_model.inverse(z, steps=i).detach().cpu().numpy()
        # print(x.shape)
        plt.figure(figsize=(6, 6))
        plt.scatter(x[:, 0], x[:, 1], s=10, c="blue", alpha=0.5)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.title(f"Transformed Prior Distribution after {i} layers")
        summary_writer.add_figure("transformation_steps", plt.gcf(), i)
        plt.close()

    # save target distribution
    x = run_sampling(4096)
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], s=10, c="blue", alpha=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Target Distribution")
    summary_writer.add_figure("target_distribution", plt.gcf())

    Config.summary_writer.close()
