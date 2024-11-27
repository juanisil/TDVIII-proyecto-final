"""Player2Vec to Player Stats Distribution Model"""

import torch
import torch.nn as nn


class JSD(nn.Module):
    """Jenson-Shannon Divergence Loss Function

    https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/13
    """

    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        """
        p: torch.tensor
            Tensor of shape (batch_size, n_classes)
        q: torch.tensor
            Tensor of shape (batch_size, n_classes)

        Returns
        -------
        torch.tensor
            Jenson-Shannon Divergence between p and q
        """

        p, q = (
            p.view(-1, p.size(-1)).log_softmax(-1),
            q.view(-1, q.size(-1)).log_softmax(-1),
        )
        m = 0.5 * (p + q)
        return 0.5 * (self.kl(m, p) + self.kl(m, q))


class custom_loss(nn.Module):
    """Custom loss function to calculate the Jensen-Shannon Divergence between the predicted and target distributions for the 5 features
    Ponderation of the JSD divergence of the mean and std of the 5 features
    """

    def __init__(self):
        super(custom_loss, self).__init__()
        self.jsd = JSD()

    def forward(self, y_pred, y_true):
        """
        y_pred: torch.tensor
            Tensor of shape (batch_size, 10)
        y_true: torch.tensor
            Tensor of shape (batch_size, 10)
        """
        
        """
            [
                0 'losses_prob_mean',
                1 'gains_prob_mean',
                2 'shots_prob_mean',
                3 'avg_pass_to_prob_mean',
                4 'avg_pass_from_prob_mean',
                5 'losses_prob_std',
                6 'gains_prob_std',
                7 'shots_prob_std',
                8 'avg_pass_to_prob_std',
                9 'avg_pass_from_prob_std'
            ]
        """

        # 0, 5 -> losses mean, std
        # 1, 6 -> gains mean, std
        # 2, 7 -> shots mean, std
        # 3, 8 -> avg pass to mean, std
        # 4, 9 -> avg pass from mean, std

        loss = 0
        for i in range(5):
            loss += self.jsd(y_pred[:, i], y_true[:, i]) + self.jsd(
                y_pred[:, i + 5], y_true[:, i + 5]
            )
        return loss


class p2v_dist_model(nn.Module):
    """Player2Vec to Player Stats Distribution Model"""

    def __init__(self, input_size=3, output_size=10):
        super(p2v_dist_model, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        """
        x: torch.tensor
            Tensor of shape (batch_size, input_size)

        Returns
        -------
        torch.tensor
            Tensor of shape (batch_size, output_size)
        """

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_epochs=10000,
        early_stopping=True,
        optimizer=None,
        scheduler=None,
        criterion=None,
        callbacks=None,
    ):
        """
        X_train: torch.tensor
            Tensor of shape (n_samples, input_size)
        y_train: torch.tensor
            Tensor of shape (n_samples, output_size)
        X_test: torch.tensor
            Tensor of shape (n_samples, input_size)
        y_test: torch.tensor
            Tensor of shape (n_samples, output_size)
        n_epochs: int
            Number of epochs to train the model
        early_stopping: bool
            Whether to stop training early if the test loss does not improve
        optimizer: torch.optim.Optimizer
            Optimizer to use for training
        scheduler: torch.optim.lr_scheduler
            Learning rate scheduler to use for training
        callbacks: list
            List of callbacks to call after each epoch
        criterion: nn.Module
            Loss function to use for training
        """

        if optimizer is None:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
            )

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=100, gamma=0.1
            )

        if criterion is None:
            criterion = custom_loss()

        train_losses = []
        test_losses = []

        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if scheduler is not None:
                scheduler.step(loss)

            self.eval()
            with torch.no_grad():
                y_pred = self(X_test)
                loss = criterion(y_pred, y_test)
                test_losses.append(loss.item())

                # early stopping
                if (
                    early_stopping
                    and len(test_losses) > 10
                    and test_losses[-10] < test_losses[-1]
                ):
                    break

            for callback in callbacks:
                callback(self, epoch, train_losses, test_losses)

        return train_losses, test_losses
