from kedro.pipeline import Pipeline, node, pipeline

from .feature_engineering_nodes import create_features

# from feature_factory import create_features


def create_pipeline() -> Pipeline:
    return pipeline(
        [
            # TODO: Remove below node once create_features node is ready
            # node(
            #     func=create_derived_features,
            #     inputs=["df_merged_clean", "params:derived_features"],
            #     outputs="df_merged_with_derived_features",
            #     name="create_derived_features",
            #     tags="feature_eng",
            # ),
            node(
                func=create_features,
                inputs={
                    "params": "params:feature_eng",
                    "data": "df_merged_outlier_clean",
                    "user_inputs_form_conversion": "user_input_form_conversion",
                },
                outputs="df_merged_with_derived_features",
                name="create_derived_features",
                tags="feature_factory",
            ),
        ]
    )
