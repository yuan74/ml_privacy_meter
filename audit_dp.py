import time
from pathlib import Path

import numpy as np
import torch.utils.data
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Subset

from attacks import tune_offline_a, run_rmia, run_loss
from visualize import plot_roc, plot_roc_log
from auditor import get_eps_audit

from matplotlib import pyplot as plt
import textwrap



def plot_eps_vs_num_guesses(eps_list, correct_num_list, k_neg_list, k_pos_list, total_num, path):
    """Function to get the auditing performance versus number of guesses plot

    Args:
        eps_list (list or ndarray): List of audited eps values
        correct_num_list (list or ndarray): List of number of correct guesses
        k_neg_list (list or ndarray): List of positive guesses
        k_pos_list (list or ndarray): List of negative guesses
        total_num (int): Total number of samples
        path (str): Folder for saving the auditing performance plot
    """
    fig, ax = plt.subplots(1, 1)
    num_guesses_grid = np.array(k_neg_list) + total_num - np.array(k_pos_list)
    ax.scatter(num_guesses_grid, correct_num_list/num_guesses_grid,
        color = '#FF9999', alpha=0.6, label=r'Inference Accuracy', s = 80)
    ax.scatter(num_guesses_grid, eps_list,
        color = '#66B2FF', alpha=0.6, label=r'$EPS LB$', s = 80)
    ax.set_xlabel(r'number of guesses')
    ax.set_ylim(0, 1)
    plt.legend(fontsize=10)

    min_interval_idx = np.argmax(eps_list)
    t = f"k_neg={k_neg_list[min_interval_idx]} and k_pos={k_pos_list[min_interval_idx]} enables the highest audited EPS LB: num of guesses is {num_guesses_grid[min_interval_idx]}, EPS LB is {eps_list[min_interval_idx]}"
    tt = textwrap.fill(t, width = 70)
    plt.text(num_guesses_grid[len(num_guesses_grid)//2], -0.2, tt, ha='center', va='top')
    # plt.text(f"{min_interval_idx}-th choice of intervals has optimal objective: num of guesses is {num_samples_grid[min_interval_idx]}, f-g value is {f_minus_g_grid[min_interval_idx]}, translated eps is {logit(f_minus_g_grid[min_interval_idx])}, hat eps is {hat_eps_grid[min_interval_idx]}")
    # plt.text(f"{min_interval_idx_audit}-th choice of intervals has optimal audited lower bound: num of guesses is {num_samples_grid[min_interval_idx_audit]}, f-g value is {f_minus_g_grid[min_interval_idx_audit]}, translated eps is {logit(f_minus_g_grid[min_interval_idx_audit])}, hat eps is {hat_eps_grid[min_interval_idx_audit]}")
    # plt.text(f"{member_text} intervals: {member_intervals[min_interval_idx]}")
    # plt.text(f"{non_member_text} intervals (audit): {non_member_intervals[min_interval_idx_audit]}")
    
   
    plt.savefig(path, bbox_inches = 'tight')
    
    plt.close()



def compute_abstain_attack_results(mia_scores, target_memberships, delta=0, p_value=0.05):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    mia_scores = mia_scores.ravel()
    target_memberships = target_memberships.ravel()
    sorted_idx = np.argsort(mia_scores)
    mia_scores = mia_scores[sorted_idx]
    target_memberships = target_memberships[sorted_idx]
    step_size = int(np.sqrt(len(target_memberships.ravel())))
    assert(step_size>=1)
    k_neg_k_pos_list = sum([[(k_neg, k_pos) for k_neg in range(0, k_pos, step_size)] for k_pos in range(0, len(target_memberships.ravel()), step_size)], [])
    correct_num_list = [(1-target_memberships[:k_neg]).sum() +  target_memberships[k_pos:].sum() for (k_neg, k_pos) in k_neg_k_pos_list]
    eps_list = [get_eps_audit(len(target_memberships), k_neg + len(target_memberships) - k_pos, correct_num, delta, p_value) for ((k_neg, k_pos), correct_num) in zip(k_neg_k_pos_list, correct_num_list)]
    k_neg_k_pos_idx = np.argmax(eps_list)
    (k_neg_opt, k_pos_opt) = k_neg_k_pos_list[k_neg_k_pos_idx]
    eps_opt = eps_list[k_neg_k_pos_idx]
    correct_num_opt = correct_num_list[k_neg_k_pos_idx]
    

    return {
        "k_neg": [k_neg_k_pos_list[i][0] for i in range(len(k_neg_k_pos_list))],
        "k_pos": [k_neg_k_pos_list[i][1] for i in range(len(k_neg_k_pos_list))],
        "eps": eps_list,
        "correct_num": correct_num_list,
        "eps_opt": eps_opt,
        "k_neg_opt": k_neg_opt,
        "k_pos_opt": k_pos_opt,
        "correct_num_opt": correct_num_opt,
        "total_num": len(target_memberships),
        "delta": delta,
        "p_value": p_value,
    }


def compute_abstain_attack_results_for_k_pos_k_neg(mia_scores, target_memberships, k_pos, k_neg, delta=0, p_value=0.05):
    """
    Compute attack results (TPR-FPR curve, AUC, etc.) based on MIA scores and membership of samples.

    Args:
        mia_scores (np.array): MIA score computed by the attack.
        target_memberships (np.array): Membership of samples in the training set of target model.

    Returns:
        dict: Dictionary of results, including fpr and tpr list, AUC, TPR at 1%, 0.1% and 0% FPR.
    """
    mia_scores = mia_scores.ravel()
    target_memberships = target_memberships.ravel()
    sorted_idx = np.argsort(mia_scores)
    mia_scores = mia_scores[sorted_idx]
    target_memberships = target_memberships[sorted_idx]
    correct_num  = (1-target_memberships[:k_neg]).sum() +  target_memberships[k_pos:].sum() 
    eps = get_eps_audit(len(target_memberships), k_neg + len(target_memberships) - k_pos, correct_num, delta, p_value)

    return {
        "k_neg": k_neg,
        "k_pos": k_pos,
        "correct_num": correct_num,
        "eps": eps,
        "total_num": len(target_memberships),
        "delta": delta,
        "p_value": p_value,
    }

def get_all_dp_audit_results(report_dir, mia_score_list, membership_list, logger):
    """
    Generate and save ROC plots for attacking multiple models by aggregating all scores and membership labels.

    Args:
        report_dir (str): Folder for saving the ROC plots.
        mia_score_list (list): List of MIA scores for each target model.
        membership_list (list): List of membership labels of each target model.
        logger (logging.Logger): Logger object for the current run.
    """

    mia_scores = np.concatenate(mia_score_list)
    target_memberships = np.concatenate(membership_list)

    attack_dp_result = compute_abstain_attack_results(mia_scores, target_memberships)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Best One Run DP Auditing Results: EPS Lower Bound %.4f under DELTA %.2e and P_VALUE %.2f (%d correct out of %d guesses, k_neg=%d and k_pos=%d",
        attack_dp_result["eps_opt"],
        attack_dp_result["delta"],
        attack_dp_result["p_value"],
        attack_dp_result["correct_num_opt"],
        attack_dp_result["k_neg_opt"] + attack_dp_result["k_pos_opt"],
        attack_dp_result["k_neg_opt"],
        attack_dp_result["k_pos_opt"]
    )

    plot_eps_vs_num_guesses(
        attack_dp_result["eps"],
        attack_dp_result["correct_num"],
        attack_dp_result["k_neg"],
        attack_dp_result["k_pos"],
        attack_dp_result["total_num"],
        f"{report_dir}/dp_audit_average.png",
    )

    np.savez(
        f"{report_dir}/attack_result_average_dp",
        eps=attack_dp_result["eps"],
        correct_num=attack_dp_result["correct_num"],
        k_neg=attack_dp_result["k_neg"],
        k_pos=attack_dp_result["k_pos"],
        total_num=attack_dp_result["total_num"],
        scores=mia_scores.ravel(),
        memberships=target_memberships.ravel(),
    )



def get_dp_audit_results_for_k_pos_k_neg(report_dir, mia_score_list, membership_list, logger, k_pos, k_neg):
    """
    Generate and save ROC plots for attacking multiple models by aggregating all scores and membership labels.

    Args:
        report_dir (str): Folder for saving the ROC plots.
        mia_score_list (list): List of MIA scores for each target model.
        membership_list (list): List of membership labels of each target model.
        logger (logging.Logger): Logger object for the current run.
    """

    mia_scores = np.concatenate(mia_score_list)
    target_memberships = np.concatenate(membership_list)

    attack_dp_result = compute_abstain_attack_results_for_k_pos_k_neg(mia_scores, target_memberships, k_pos, k_neg)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "One Run DP Auditing Results: EPS Lower Bound %.4f under DELTA %.2e and P_VALUE %.2f (%d correct out of %d guesses)",
        attack_dp_result["eps"],
        attack_dp_result["delta"],
        attack_dp_result["p_value"],
        attack_dp_result["correct_num"],
        attack_dp_result["k_neg"] + attack_dp_result["total_num"] - attack_dp_result["k_pos"],
    )


# todo: add canary data (mislabelled data)