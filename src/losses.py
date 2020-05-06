import torch


def weight_decay(weights: list):
    # reshaped = [weight.reshape(-1) for weight in weights]
    tests = [torch.dot(weight.reshape(-1), weight.reshape(-1)) for weight in weights]
    weight_norms = torch.stack(tests, dim=-1)
    return torch.sum(weight_norms)


def latent_kl(prior_mean, posterior_mean):
    """

    :param prior_mean:
    :param posterior_mean:
    :return:
    """
    kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
    kl = torch.sum(kl, dim=[1, 2, 3])
    kl = torch.mean(kl)

    return kl


def aggregate_kl_loss(prior_means, posterior_means):
    kl_loss = torch.sum(
        torch.cat(
            [
                latent_kl(p, q).unsqueeze(dim=-1)
                for p, q in zip(prior_means, posterior_means)
            ],
            dim=-1,
        )
    )
    return kl_loss


def vgg_loss(custom_vgg, target, pred):
    """

    :param custom_vgg:
    :param target:
    :param pred:
    :return:
    """
    target_feats = custom_vgg(target)
    pred_feats = custom_vgg(pred)

    loss = torch.cat(
        [
            FLAGS.vgg_feat_weights[i] * torch.mean(torch.abs(tf - pf)).unsqueeze(dim=-1)
            for i, (tf, pf) in enumerate(zip(target_feats, pred_feats))
        ],
        dim=-1,
    )

    loss = torch.sum(loss)
    return loss
