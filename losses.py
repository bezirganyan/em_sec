import numpy as np
import torch
import torch.nn.functional as F
from torch import digamma


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    if not useKL:
        return loglikelihood

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    return torch.mean(loss)


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum


def get_evidential_loss(evidence, target, epoch_num, num_classes, annealing_step, device, targets_one_hot=False,
                        uncertainty_calibration=False):
    if not targets_one_hot:
        target = F.one_hot(target, num_classes)
    alpha_a = evidence + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    if not uncertainty_calibration:
        return loss_acc
    calibration_loss = uncertainty_calibration_loss(evidence, target, 0.001, epoch_num, 200)
    return loss_acc + calibration_loss


def belief_matching(alphas, ys):
    prior = 1.
    coeff = 0.1
    # alphas = torch.exp(alphas)
    betas = prior * torch.ones_like(alphas)
    ys = ys.long()
    a_ans = torch.gather(alphas, -1, ys.unsqueeze(-1)).squeeze(-1)
    a_zero = torch.sum(alphas, -1)
    ll_loss = digamma(a_ans) - digamma(a_zero)

    loss1 = torch.lgamma(a_zero) - torch.sum(torch.lgamma(alphas), -1)
    loss2 = torch.sum(
        (alphas - betas) * (digamma(alphas) - digamma(a_zero.unsqueeze(-1))),
        -1)
    kl_loss = loss1 + loss2

    loss = (coeff * kl_loss - ll_loss).mean()
    return loss


def get_bm_loss(alphas, alpha_a, target):
    loss_acc = belief_matching(alpha_a, target)
    for v in range(len(alphas)):
        alpha = alphas[v] + 1
        loss_acc += belief_matching(alpha, target)
    loss_acc = loss_acc / (len(alphas) + 1)
    # loss = loss_acc + gamma * get_dc_loss(evidences, device)
    return loss_acc


def soft_fbeta(probs, targets, beta=1.0, epsilon=1e-8):
    batch_size, num_classes = probs.shape

    soft_includes = []
    for b in range(batch_size):
        p_correct = probs[b][targets[b].bool()]  # predicted probs for correct classes
        if len(p_correct) == 0:
            soft_includes.append(probs.new_tensor([0.0]))
        else:
            prod_not_included = torch.prod(1 - p_correct)
            soft_includes.append(1.0 - prod_not_included)
    soft_includes = torch.stack(soft_includes, dim=0)
    cardinalities = probs.sum(dim=1)

    fbeta_vals = (2.0 + beta ** 2) / (cardinalities + beta ** 2 + epsilon)
    fbeta_vals = fbeta_vals * soft_includes

    return fbeta_vals


def ava_edl_criterion(
    B_alpha, B_beta, targets,
    *,                      # force kwargs after this
    discount: torch.Tensor | None = None,  # shape (C, C)
    lambda_fbeta: float = 1.0,
):
    pos_term = torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha)
    neg_term = torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)

    pos_loss = targets * pos_term
    neg_loss = (1.0 - targets) * neg_term

    # take from each row the element corresponding to the class

    if discount is None:
        neg_loss = neg_loss * lambda_fbeta
    else:
        discounts_per_class = discount[targets.argmax(dim=1)]
        neg_loss = neg_loss * discounts_per_class
    loss_cls = (pos_loss + neg_loss).mean()
    return loss_cls #+ reg


def edl_hyperloss(func, y, alpha, hyperset_soft_size, epoch_num, num_classes, annealing_step, device, useKL=True,
                  lmda=0.0):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    # normalizer of ones for each class, but (1 / hyperset_soft_size) for the last class
    hyperset_normalizer = torch.ones_like(y)
    hyperset_normalizer[:, -1] = torch.maximum(hyperset_soft_size * lmda, torch.ones_like(y)[:, -1])
    # hyperset_normalizer[:, -1] = 5
    A = torch.sum(y * (func(S) - func(alpha)) * hyperset_normalizer, dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes + 1, device=device)
    return A + kl_div


def edl_digamma_hyperloss(alpha, target, hyperset_soft_size, epoch_num, num_classes, annealing_step, device):
    loss = edl_hyperloss(torch.digamma, target, alpha, hyperset_soft_size, epoch_num, num_classes, annealing_step,
                         device)
    return torch.mean(loss)


def get_subjective_constraint(evidence, multilabel_probs):
    multinomial_evidence = evidence[:, :-1]
    hyper_evidence = evidence[:, -1]
    beliefs = evidence / (evidence + 1).sum(dim=1, keepdim=True)
    uncertainty = evidence.shape[1] / (evidence + 1).sum(dim=1)

    focal_uncertainty = uncertainty / evidence.shape[1]
    hyperset = torch.sigmoid(1000 * (multilabel_probs - 0.5))
    vague_belief_mass = hyperset * hyper_evidence.unsqueeze(-1) / hyperset.sum(dim=1, keepdim=True)

    projected_probability = vague_belief_mass + focal_uncertainty.unsqueeze(-1) + multinomial_evidence

    return projected_probability


def get_evidential_hyperloss(evidence, multilabel_probs, target, epoch_num, num_classes, annealing_step, device,
                             beta=1):
    # pp = get_subjective_constraint(evidence, multilabel_probs)
    hyperset = (multilabel_probs > 0.5).int()
    hyperset_corrects = ((hyperset & target.int()).sum(dim=1) > 0).unsqueeze(-1)
    # target = torch.cat((target, torch.ones(target.shape[0], 1).to(device) * hyperset_corrects), dim=1)
    target = torch.cat((target, torch.ones(target.shape[0], 1).to(device)), dim=1)
    hyperset_soft_size = torch.sigmoid(1000 * (multilabel_probs - 0.5)).sum(dim=1)

    gfb = (1 + beta ** 2) / (hyperset_soft_size + beta ** 2)
    discount = torch.ones(target.shape[0], target.shape[1] - 1).to(device)
    discount = torch.cat((discount, gfb.unsqueeze(-1)), dim=1)
    # target = target * discount
    alpha_a = evidence + 1
    loss_acc = edl_digamma_hyperloss(alpha_a, target, hyperset_soft_size, epoch_num, num_classes, annealing_step,
                                     device)
    return loss_acc


def get_fbeta_loss(evidence, multilabel_probs, beta=1):
    hyperset_soft_size = torch.sigmoid(1000 * (multilabel_probs - 0.5)).sum(dim=1)
    soft_sizes = torch.ones(hyperset_soft_size.shape).to(evidence.device)
    hyper_choices = (evidence.argmax(dim=1) == evidence.shape[1] - 1)
    soft_sizes[hyper_choices] *= hyperset_soft_size[hyper_choices]

    gfb = (1 + beta ** 2) / (soft_sizes + beta ** 2)
    return -torch.log(gfb).mean()


def get_equivalence_loss(multinomial_evidence, hyper_evidence):
    beliefs_m = multinomial_evidence / (multinomial_evidence + 1).sum(dim=1, keepdim=True)
    beliefs_h = hyper_evidence[:, :-1] / (hyper_evidence[:, :-1] + 1).sum(dim=1, keepdim=True)

    loss = torch.square(beliefs_m - beliefs_h).mean()

    return loss


def get_utility_loss(evidence, multilabel_probs, target, beta, device):
    set_utility = torch.ones(target.shape[0], 1).to(device)
    hyperset_soft_size = torch.sigmoid(10000 * (multilabel_probs - 0.5)).sum(dim=1)
    gfb = (1 + beta ** 2) / (hyperset_soft_size + beta ** 2)
    set_utility = set_utility * gfb.unsqueeze(-1)
    target = torch.cat((target, set_utility), dim=1)
    probs = (evidence + 1) / (evidence + 1).sum(dim=1, keepdim=True)
    utility_abs = probs * target
    utility_loss = torch.mean(utility_abs)
    # return -utility_loss
    return 0


def uncertainty_calibration_loss(evidence, target, annealing_factor, epoch_num, total_epochs):
    num_classes = evidence.shape[1]
    eps = 1e-10
    uncertainty = num_classes / (evidence + 1).sum(dim=1, keepdim=True)
    annealing_factor = torch.tensor(annealing_factor).to(evidence.device)
    probs = (evidence + 1) / (evidence + 1).sum(dim=1, keepdim=True)
    corrects = (target.argmax(dim=1) == probs.argmax(dim=1)).float().reshape(-1, 1)
    lambda_t = annealing_factor * torch.exp(-epoch_num * (torch.log(annealing_factor) / total_epochs))
    probs = torch.max(probs, dim=1).values.reshape(-1, 1)
    corr_loss = - lambda_t * probs * torch.log(1 - uncertainty + eps) * corrects
    incor_loss = (1 - lambda_t) * ((1 - probs) * torch.log(uncertainty + eps) * (1 - corrects))
    return torch.mean(corr_loss + incor_loss)