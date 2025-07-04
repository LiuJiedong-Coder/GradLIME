import quantus

class MyMetrics(object):

    # 忠实性
    def compute_Faithfulness(model, x_batch, y_batch, a_batch, type, device=None):
        if type == 'Correlation':  # 2020
            score = quantus.FaithfulnessCorrelation(
                nr_runs=100,
                subset_size=224,
                perturb_baseline="black",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                similarity_func=quantus.similarity_func.correlation_pearson,
                abs=False,
                return_aggregate=False,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'Estimate':  # 2018
            # Return faithfulness estimate scores in an one-liner - by calling the metric instance.
            score = quantus.FaithfulnessEstimate(
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                similarity_func=quantus.similarity_func.correlation_pearson,
                features_in_step=224,
                perturb_baseline="black",
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)
        elif type == 'Monotonicity':  # 2019, Arya
            # Return monotonicity scores in an one-liner - by calling the metric instance.
            score = quantus.Monotonicity(
                features_in_step=224,
                perturb_baseline="black",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'MonotonicityCorrelation':  # 2020,Nguyen
            # Return monotonicity scores in an one-liner - by calling the metric instance.
            score = quantus.MonotonicityCorrelation(
                nr_samples=10,
                features_in_step=3136,
                perturb_baseline="uniform",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                similarity_func=quantus.similarity_func.correlation_spearman,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'PixelFlipping':  # 2015, 生成224个值
            # Create the pixel-flipping experiment.
            score = quantus.PixelFlipping(
                features_in_step=224,
                perturb_baseline="black",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'Sufficiency':  # 2022
            score = quantus.Sufficiency(
                threshold=0.6,
                return_aggregate=False,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'RegionPerturbation':  # 2015, 生成regions_evaluation个
            score = quantus.RegionPerturbation(
                patch_size=14,
                regions_evaluation=1,
                perturb_baseline="uniform",
                normalise=True
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'Selectivity':  # 2018 生成15个值
            score = quantus.Selectivity(
                patch_size=56,
                perturb_baseline="black"
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'SensitivityN':  # 2018 报错
            score = quantus.SensitivityN(
                features_in_step=224,
                n_max_percentage=0.8,
                similarity_func=quantus.similarity_func.correlation_pearson,
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                perturb_baseline="uniform",
                return_aggregate=False
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'IROF':  # 2020
            score = quantus.IROF(
                segmentation_method="slic",
                perturb_baseline="mean",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                return_aggregate=False
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'Infidelity':  # 2019
            score = quantus.Infidelity(
                perturb_baseline="uniform",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                n_perturb_samples=5,
                perturb_patch_sizes=[56],
                display_progressbar=False
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'ROAD':  # 2022, 生成有0 有1 的值
            # Return ROAD scores in an one-liner - by calling the metric instance.
            score = quantus.ROAD(
                noise=0.01,
                perturb_func=quantus.perturb_func.noisy_linear_imputation,
                percentages=list(range(1, 50, 2)),
                display_progressbar=False,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        else:
            print('Wrong type of Faithfulness')
        return score

    # 稳健性，鲁棒性
    def compute_Robustness(model, x_batch, y_batch, a_batch, type, device=None):

        if type == 'LocalLipschitzEstimate':  # 2018,2019， 报错
            score = quantus.FaithfulnessCorrelation(
                nr_samples=10,
                perturb_std=0.2,
                perturb_mean=0.0,
                norm_numerator=quantus.similarity_func.distance_euclidean,
                norm_denominator=quantus.similarity_func.distance_euclidean,
                perturb_func=quantus.perturb_func.gaussian_noise,
                similarity_func=quantus.similarity_func.lipschitz_constant,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device
              )

        elif type =='MaxSensitivity':  #2019, 报错
            score =quantus.MaxSensitivity(
                nr_samples=10,
                lower_bound=0.2,
                norm_numerator=quantus.norm_func.fro_norm,
                norm_denominator=quantus.norm_func.fro_norm,
                perturb_func=quantus.perturb_func.uniform_noise,
                similarity_func=quantus.similarity_func.difference
            )(model=model,
               x_batch=x_batch,
               y_batch=y_batch,
               a_batch=a_batch,
               device=device
              )


        elif type =='AvgSensitivity':  #2019, 报错
            score =quantus.AvgSensitivity(
                nr_samples=10,
                lower_bound=0.2,
                norm_numerator=quantus.norm_func.fro_norm,
                norm_denominator=quantus.norm_func.fro_norm,
                perturb_func=quantus.perturb_func.uniform_noise,
                similarity_func=quantus.similarity_func.difference,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device
              )

        elif type == 'Continuity':   #2017,报错
            score = quantus.Continuity(
                patch_size=56,
                nr_steps=10,
                perturb_baseline="uniform",
                similarity_func=quantus.similarity_func.correlation_spearman,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'Consistency':   #2022 可用，生成0
            score = quantus.Consistency(
                discretise_func=quantus.discretise_func.top_n_sign,
                return_aggregate=False,
            )(model=model,
               x_batch=x_batch,
               y_batch=y_batch,
               a_batch=a_batch,
               device=device)

        elif type == 'RelativeInputStability':   #2022,报错
            score = quantus.RelativeInputStability(nr_samples=5)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                device=device
            )

        elif type == 'RelativeOutputStability':  # 2022， 报错
            score = quantus.RelativeOutputStability(nr_samples=5)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                device=device
            )

        elif type == 'RelativeRepresentationStability':  #2022， 可计算，但结果不合理
            score = quantus.RelativeRepresentationStability(nr_samples=5, layer_names=["layer4.1.conv2"])(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                explain_func=quantus.explain,
                explain_func_kwargs={"method": "Lime"},   #需要指明解释方法
                device=device,
            )

        else:
            print('Wrong type of Robustness')
        return score

    # 随机性
    def compute_Randomisation(model, x_batch, y_batch, a_batch, type, device=None):

        if type == 'MPRT':    #2018报错，原名ModelParameterRandomisation
            score = quantus.MPRT(
                layer_order="bottom_up",
                similarity_func=quantus.similarity_func.correlation_spearman,
                normalise=True,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'RandomLogit':  #2020报错
            score = quantus.RandomLogit(
                num_classes=1000,
                similarity_func=quantus.similarity_func.ssim,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        else:
            print('Wrong type of Randomisation')

        return score

    # 复杂性
    def compute_Complexity(model, x_batch, y_batch, a_batch, type, device=None):
        if type == 'Sparseness':    #2018, 2019
            score = quantus.Sparseness(
            )(model=model,
               x_batch=x_batch,
               y_batch=y_batch,
               a_batch=a_batch,
               device=device)

        elif type == 'Complexity':  #2020
            score = quantus.Complexity(
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'EffectiveComplexity':  # 2020
            score = quantus.EffectiveComplexity(eps=1e-5,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        else:
            print('Wrong type of Complexity')

        return score

    # 公理性
    def compute_Axiomatic(model, x_batch, y_batch, a_batch, type, device=None):
        if type == 'Completeness':  # 2018
            score = quantus.Completeness(
                abs=False,
                disable_warnings=True,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type =='NonSensitivity':    #2020
            score = quantus.NonSensitivity(
                abs=True,
                eps=1e-5,
                n_samples=10,
                perturb_baseline="black",
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                features_in_step=6272,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)

        elif type == 'InputInvariance':   #2017报错
            score = quantus.InputInvariance(
                abs=False,
                disable_warnings=True,
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              device=device)
        else:
            print('Wrong type of Complexity')

        return score

    def compute_Localisation(model, x_batch, y_batch, a_batch, s_batch, type, device=None):
        if type == 'PointingGame':
            score = quantus.PointingGame(  #2018
            )(model=model,
               x_batch=x_batch,
               y_batch=y_batch,
               a_batch=a_batch,
               s_batch=s_batch,
               device=device)

        elif type == 'AttributionLocalisation':   #2020
            # Return attribution localisation scores in an one-liner - by calling the metric instance.
            score = quantus.AttributionLocalisation(
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              s_batch=s_batch,
              device=device)
        elif type =='TKI':   #2021
            # Return tki scores in an one-liner - by calling the metric instance.
            score = quantus.TopKIntersection(
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              s_batch=s_batch,
              device=device)
        elif type == 'RelevanceRankAccuracy':  #2021
            # Return relevane rank accuracy scores in an one-liner - by calling the metric instance.
            score = quantus.RelevanceRankAccuracy(
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              s_batch=s_batch,
              device=device)
        elif type == 'RelevanceMassAccuracy':   #2021
            # Return relevane rank accuracy scores in an one-liner - by calling the metric instance.
            score = quantus.RelevanceMassAccuracy(
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              s_batch=s_batch,
              device=device)

        elif type == 'AUC':   #2021
            # Return relevane mass accuracy scores in an one-liner - by calling the metric instance.
            score = quantus.AUC(
            )(model=model,
              x_batch=x_batch,
              y_batch=y_batch,
              a_batch=a_batch,
              s_batch=s_batch,
              device=device)
        else:
            print('Wrong type of Localisation')

        return score

