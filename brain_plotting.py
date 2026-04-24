import os
import numpy as np
import pickle as pkl
from nilearn import plotting, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Any, Optional
from tqdm import tqdm
from transformers import AutoConfig

def load_pickle(path: str) -> Any:
    with open(path, 'rb') as fin:
        data = pkl.load(fin)
    return data

class BrainPlotter:
    """A class to handle brain surface visualization and correlation plots."""

    @staticmethod
    def plot_left_hemisphere_correlations(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Left Hemisphere Correlations",
        only_significant: bool = True,
    ) -> plt.Figure:
        """
        Plot left hemisphere correlations.

        Args:
            correlations: Array of correlation values (length should match number of vertices)
            significant_mask: Boolean mask indicating significant vertices
            title: Plot title
            only_significant: If True, only plot significant correlations

        Returns:
            matplotlib Figure object
        """
        # Apply significance mask if requested
        masked_correlations = correlations.copy()
        if only_significant:
            masked_correlations[~significant_mask] = np.nan

        # Get fsaverage surface
        fsaverage = datasets.fetch_surf_fsaverage()

        # Create figure
        # fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "3d"})
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            masked_correlations[:10242],  # Left hemisphere has 10242 vertices
            hemi="left",
            view="lateral",
            colorbar=True,
            title=title,
            vmin=-1,
            vmax=1,
            cmap='cold_hot',
        )

        # plt.tight_layout()

    @staticmethod
    def plot_left_right_hemisphere_correlations(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Left and Right Hemisphere Correlations",
        only_significant: bool = True,
        cmap: str = 'cold_hot',
        vmin: float = -1,
        vmax: float = 1,
    ) -> plt.Figure:
        """
        Plot left and right hemisphere correlations.

        Args:
            correlations: Array of correlation values (length should match number of vertices)
            significant_mask: Boolean mask indicating significant vertices
            title: Plot title
            only_significant: If True, only plot significant correlations
        Returns:
            matplotlib Figure object
        """
        # Apply significance mask if requested
        masked_correlations = correlations.copy()
        if only_significant:
            masked_correlations[~significant_mask] = np.nan

        # Get fsaverage surface
        fsaverage = datasets.fetch_surf_fsaverage()

        # Create figure
        fig = plt.figure(figsize=(10, 6))

        # Plot left hemisphere
        ax1 = fig.add_subplot(121, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            masked_correlations[:10242],  # Left hemisphere has 10242 vertices
            hemi="left",
            view="lateral",
            colorbar=True,
            title="Left Hemisphere",
            axes=ax1,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        # Plot right hemisphere
        ax2 = fig.add_subplot(122, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            masked_correlations[10242:],  # Right hemisphere
            hemi="right",
            view="lateral",
            colorbar=True,
            title="Right Hemisphere",
            axes=ax2,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        plt.suptitle(title, fontsize=16)
        # plt.tight_layout()

        return fig

    @staticmethod
    def plot_surface_correlations(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Significant Prediction Correlations",
        only_significant: bool = True,
        is_volume: bool = False,
        vmin: float = None,
        vmax: float = None,
    ) -> Optional[plt.Figure]:
        """
        Plot correlations on brain surface.

        Args:
            correlations: Array of correlation values (length should match number of vertices)
            significant_mask: Boolean mask indicating significant vertices
            title: Plot title
            only_significant: If True, only plot significant correlations
            is_volume: Whether the data is volume data (if True, surface plotting is skipped)

        Returns:
            matplotlib Figure object or None if using volume data
        """
        if is_volume:
            print("Skipping surface plotting for volume data")
            return None

        # Get fsaverage surface
        fsaverage = datasets.fetch_surf_fsaverage()

        # Apply significance mask if requested
        masked_correlations = correlations.copy()
        if only_significant:
            masked_correlations[~significant_mask] = np.nan

        # Split correlations into left and right hemisphere
        n_vertices_per_hemi = 10242
        left_correlations = masked_correlations[:n_vertices_per_hemi]
        right_correlations = masked_correlations[
            n_vertices_per_hemi : 2 * n_vertices_per_hemi
        ]

        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 10))

        # Plot left hemisphere
        ax1 = fig.add_subplot(231, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            left_correlations,
            hemi="left",
            view="lateral",
            colorbar=True,
            title="Left Lateral",
            axes=ax1,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=True,
        )

        ax2 = fig.add_subplot(232, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            left_correlations,
            hemi="left",
            view="medial",
            colorbar=True,
            title="Left Medial",
            axes=ax2,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=True,
        )

        # Plot right hemisphere
        ax3 = fig.add_subplot(234, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            right_correlations,
            hemi="right",
            view="lateral",
            colorbar=True,
            title="Right Lateral",
            axes=ax3,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=True,
        )

        ax4 = fig.add_subplot(235, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            right_correlations,
            hemi="right",
            view="medial",
            colorbar=True,
            title="Right Medial",
            axes=ax4,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=True,
        )

        plt.suptitle(title, fontsize=16)
        # plt.tight_layout()

        return fig

    @staticmethod
    def plot_all_correlations_histogram(
        correlations: np.ndarray, title: str = "All Correlations Distribution"
    ) -> plt.Figure:
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.histplot(
            correlations, bins=100, color="blue", label="All", kde=True, stat="density"
        )
        plt.legend()
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        return fig

    @staticmethod
    def plot_significant_correlations_histogram(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Significant Correlations Distribution",
    ) -> plt.Figure:
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sns.histplot(
            correlations[significant_mask],
            bins=100,
            color="green",
            label="Significant",
            kde=True,
            stat="density",
        )
        plt.legend()
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        return fig

    @staticmethod
    def log_plots_to_wandb(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        prefix: str = "",
        is_volume: bool = False,
    ):
        """
        Log brain surface plots and correlation histogram to wandb.

        Args:
            correlations: Array of correlation values
            significant_mask: Boolean mask for significant correlations
            prefix: Prefix for wandb log keys
            is_volume: Whether the data is volume data
        """
        # Create and log brain surface plots only for surface data
        if not is_volume:
            fig_significant = BrainPlotter.plot_surface_correlations(
                correlations,
                significant_mask,
                title="Significant Prediction Correlations",
                only_significant=True,
                is_volume=is_volume,
            )
            if fig_significant is not None:
                wandb.log(
                    {f"{prefix}brain_surface_significant": wandb.Image(fig_significant)}
                )
                plt.close(fig_significant)

            fig_all_surface = BrainPlotter.plot_surface_correlations(
                correlations,
                significant_mask,
                title="All Prediction Correlations",
                only_significant=False,
                is_volume=is_volume,
            )
            if fig_all_surface is not None:
                wandb.log({f"{prefix}brain_surface_all": wandb.Image(fig_all_surface)})
                plt.close(fig_all_surface)

        # Plot and log all correlations histogram
        fig_all = BrainPlotter.plot_all_correlations_histogram(
            correlations, title="All Correlations Distribution"
        )
        wandb.log({f"{prefix}correlation_histogram_all": wandb.Image(fig_all)})
        plt.close(fig_all)

        # Plot and log only significant correlations histogram
        fig_sig = BrainPlotter.plot_significant_correlations_histogram(
            correlations,
            significant_mask,
            title="Significant Correlations Distribution",
        )
        wandb.log({f"{prefix}correlation_histogram_significant": wandb.Image(fig_sig)})
        plt.close(fig_sig)

        # Log raw histogram data for all
        wandb.log(
            {f"{prefix}correlation_histogram_data_all": wandb.Histogram(correlations)}
        )
        # Log raw histogram data for significant
        wandb.log(
            {
                f"{prefix}correlation_histogram_data_significant": wandb.Histogram(
                    correlations[significant_mask]
                )
            }
        )

        # Print statistics
        sig_corrs = correlations[significant_mask]
        print(
            f"Significant correlation range: {sig_corrs.min():.3f} to {sig_corrs.max():.3f}"
        )
        print(f"Mean significant correlation: {sig_corrs.mean():.3f}")
        print(f"Number of significant vertices: {len(sig_corrs)}")
        print(
            f"Percentage significant: {100 * len(sig_corrs) / len(correlations):.1f}%"
        )


if __name__ == "__main__":

    # checkpoints = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]
    # model_name = "EleutherAI/pythia-410m"
    # model_base_name = os.path.basename(model_name)


    # model_name = "Qwen/Qwen3-14B"
    # model_name = "Qwen/Qwen2.5-Omni-7B"
    # model_name = "Qwen/Qwen3-32B"
    # model_name = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    # # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_base_name = os.path.basename(model_name)

    # model_config = AutoConfig.from_pretrained(model_name)
    # if "Omni" in model_name:
    #     model_config = model_config.get_text_config()
        
    # num_layers = model_config.num_hidden_layers

    # # for ckpt in tqdm(reversed(checkpoints)):
    #     # savepath = f"model={model_base_name}_step={ckpt}_dataset=narratives_context={context_size}"
    # savepath = f"model={model_base_name}_dataset=lebel_pooling=last_layer=upproj"
    # savepath = f"model={model_base_name}_dataset=lebel_pooling=last_layer=block"
    # # savepath = f"model={model_base_name}_dataset=lebel_with=instruction_pooling=last_layer=block" # text-only
    # savepath = f"model={model_base_name}_dataset=lebel_modality=speech_with=instruction_pooling=last_layer=block" # speech-only
    # path = f"outputs/metrics_{savepath}"

    # # if not os.path.exists(path):
    # #     print(f">> Skipping {model_base_name} {ckpt}")
    # #     continue

    # modality = "speech" 

    # voxel_ceilings = load_pickle("outputs/lebel_voxel_ceilings_splits=500.pkl")

    # if modality in ["text", "speech"]:
        
    #     if modality == "text":
    #         layer_indices = np.arange(32, 32 + 48)  # For Qwen3-Omni models
    #     else:
    #         layer_indices = np.arange(32 + 48)  # For Qwen3-Omni models

    #     model_correlations = []
    #     model_significance_mask = []
    #     for layer_index in layer_indices:
    #         layer_path = savepath + f"-{layer_index}"
    #         path = f"outputs/metrics_{layer_path}"

    #         with open(path, "rb") as f:
    #             metrics = pkl.load(f) 

    #         model_correlations.append(metrics[0]["correlations"])
    #         model_significance_mask.append(metrics[0]["significant_mask"])

    #     model_correlations = np.stack(model_correlations)
    #     model_significance_mask = np.stack(model_significance_mask)

    #     savepath += f"_modality={modality}"
        
    # else:

    #     with open(path, "rb") as f:
    #         metrics = pkl.load(f) 

    #     layer_with_max_score = np.argmax([metric["n_significant"] for metric in metrics])
    #     print(f">> Processing {model_base_name}| Layer: {num_layers-layer_with_max_score}")

    #     # metrics = metrics[layer_with_max_score]

    #     model_correlations = np.stack([layer_metrics["correlations"] for layer_metrics in metrics])
    #     model_significance_mask = np.stack([layer_metrics["significant_mask"] for layer_metrics in metrics])

    #     savepath += "_modality=text-speech"

    # # num_significant = metrics["n_significant"]
    # # correlations = np.array(metrics["correlations"])
    # correlations = np.max(model_correlations, axis=0)
    # normalized_correlations = correlations 
    # # normalized_correlations = correlations / voxel_ceilings
    # # np.clip(normalized_correlations, -1, 1, out=normalized_correlations)

    # # detect outliers
    # # outlier_threshold = 1  # Z-score threshold for outlier detection
    # # z_scores = (normalized_correlations - np.mean(normalized_correlations)) / np.std(normalized_correlations)

    # # percentile_99 = np.percentile(normalized_correlations, 99)
    # # print(f"99th percentile of normalized correlations: {percentile_99:.3f}")
    # # percentile_1 = np.percentile(normalized_correlations, 1)
    # # print(f"1st percentile of normalized correlations: {percentile_1:.3f}")

    # # positive_outliers = normalized_correlations > percentile_99
    # # normalized_correlations[positive_outliers] = 1
    # # negative_outliers = normalized_correlations < percentile_1
    # # normalized_correlations[negative_outliers] = -1

    # # print(f"Number of positive outliers set to 1: {positive_outliers.sum()}")
    # # print(f"Number of negative outliers set to -1: {negative_outliers.sum()}")


    # # layer_correlations = model_correlations[layer_with_max_score]
    # # normalized_layer_correlations = layer_correlations / voxel_ceilings

    # correlations_indices = np.argmax(model_correlations, axis=0)

    # significance_mask = np.take_along_axis(model_significance_mask, correlations_indices[np.newaxis,:], axis=0).squeeze()

    # num_significant = significance_mask.sum()
    # max_corr = normalized_correlations.max()
    # min_corr = normalized_correlations.min()
    # mean_corr = normalized_correlations.mean()
    # median_corr = np.median(normalized_correlations)

    # # number_of_tokens = ckpt * 2_000_000
    # # num_tokens_millify = millify(number_of_tokens, precision=0)

    # print(f"N: {num_significant} | Max correlation: {max_corr:.3f} | Min correlation: {min_corr:.3f} | Median correlation: {median_corr:.3f} | Mean correlation: {mean_corr:.3f}")
    #results = load_pickle("/mnt/alphaidz/brain-score-language/brainscore_results_lebel_average.pkl")
    results = load_pickle("/mnt/alphaidz/brain-score-language/brainscore_results_layer9.pkl")
    neuroid_ids = results['neuroid_ids']
    original_order = np.array([int(nid.split('.')[-1]) for nid in neuroid_ids])
    reordered = np.zeros(len(original_order))
    reordered[original_order] = results['correlations']
    model_correlations = reordered

    model_base_name = "Apertus-8B"
    #model_correlations = load_pickle("/mnt/alphaidz/litcoder_core/results/run_20260412_180820_41710949/metrics.pkl").attrs['raw'].attrs['raw'].values
    #model_correlations = np.array(model_correlations) #.mean(axis=0)  # Average across k-folds if needed
    #model_correlations = load_pickle("/mnt/alphaidz/litcoder_core/results/run_20260412_180820_41710949/metrics.pkl")['correlations']
    #model_correlations = load_pickle("/mnt/alphaidz/brain-score-language/brainscore_results_lebel.pkl")["correlations"]
    model_significance_mask = np.ones_like(model_correlations, dtype=bool)  # Assuming all correlations are significant for this example
    figure = BrainPlotter.plot_left_right_hemisphere_correlations(
        np.array(model_correlations),
        np.array(model_significance_mask),
        title=f"{model_base_name}",
        only_significant=True,
        # is_volume=False,
        vmin=-0.1,
        vmax=0.4,
    )
    print(results['mean_score'])
    plt.tight_layout()
    plt.savefig(f"layer9_lebel_average.png")
    plt.cla()
    plt.clf()
    plt.close(figure)

    # print(f"Indices Shape: {correlations_indices.shape} | Significance Mask Shape: {significance_mask.shape}")

    # figure = BrainPlotter.plot_left_right_hemisphere_correlations(
    #     np.array(correlations_indices),
    #     np.array(significance_mask),
    #     title=f"{model_base_name}\nMedian: {median_corr:.3f} | Mean: {mean_corr:.3f} | Max: {max_corr:.3f}",
    #     only_significant=False,
    #     cmap="Spectral",
    #     vmin=0,
    #     vmax=80 - 1 if modality != "text" else 48 - 1,
    # )

    # plt.tight_layout()
    # plt.savefig(f"outputs/brainplot_layer_max_{savepath}.png")
    # plt.cla()
    # plt.clf()
    # plt.close(figure)
        
    # stimulus_index = 1000
    # max_value = data["brain_data"][:50][:10242].max()
    # min_value = data["brain_data"][:50][:10242].min()
    # for stimulus_index in tqdm(range(len(data["stimulus"]))):

    #     BrainPlotter.plot_left_hemisphere_correlations(
    #         data["brain_data"][stimulus_index],
    #         np.ones_like(data["brain_data"][stimulus_index], dtype=bool),
    #         title=f"Stimulus: {' '.join(data['stimulus'][stimulus_index])}",
    #         only_significant=True,
    #         vmin=0,
    #         vmax=1e+3,
    #     )

    #     plt.savefig(f"narratives_brain_images/{str(stimulus_index).zfill(4)}.png")
    #     plt.cla()
    #     plt.clf()
    #     plt.close()

    #     if stimulus_index == 50:
    #         break

