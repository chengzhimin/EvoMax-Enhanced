from pathlib import Path


def test_h200_config_points_to_uploaded_esm_if1_checkpoint() -> None:
    config = Path(__file__).parents[1] / "configs" / "h200.yaml"
    text = config.read_text()

    assert "esm_if: /data/run01/scwb286/esm2_deploy/weights/ESM-IF1/esm_if1_20220410.pt" in text
