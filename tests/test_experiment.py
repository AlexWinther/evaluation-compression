import json
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest
from torch import nn

from active_testing_benchmark import experiment
from active_testing_benchmark.benchmark import build_method_configs
from active_testing_benchmark.modeling.train import TrainingConfig


def test_method_registry_preserves_current_methods():
    aux = SimpleNamespace(surrogates=(nn.Identity(), nn.Identity()), proxy=nn.Identity())
    methods = build_method_configs(aux, num_classes=2)
    assert [cls.__name__ for cls, _ in methods] == [
        'UniformSampleMean', 'StratifiedSampleMean', 'StratifiedSampleMean',
        'KossenActiveTesting', 'ActiveSurrogateEstimator', 'PredictionPoweredActiveTesting',
    ]
    assert [kw['allocation'] for _, kw in methods[1:3]] == ['balanced', 'proportional']
    assert methods[3][1]['surrogate'] is aux.surrogates[0]
    assert methods[4][1]['surrogate'] == list(aux.surrogates)
    assert methods[5][1]['proxy'] is aux.proxy


@pytest.mark.parametrize('kwargs', [
    {'repetitions': 0}, {'budget': 0}, {'seed': -1},
    {'training_mode': 'unknown'}, {'eval_splits': ('test',)},
])
def test_invalid_configuration_fails_before_loading_data(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        experiment.run_experiment(experiment.ExperimentConfig(**kwargs))


def test_end_to_end_fast_experiment(tmp_path):
    root = tmp_path / 'data'
    rows = []
    for split_index, (split, size) in enumerate([('training', 20), ('validation', 4)]):
        folder = root / 'dr' / split
        folder.mkdir(parents=True)
        for i in range(size):
            image_id = f'{split_index}{i:03d}'
            Image.new('RGB', (8, 8), (30 + i, 50, 60)).save(folder / f'slo_fundus_{image_id}.jpg')
            rows.append({
                'filename': f'data_{image_id}.npz',
                'dr': 'vision threatening dr' if i % 2 else 'non-vision threatening dr',
                'gender': 'female' if i % 2 else 'male',
            })
    pd.DataFrame(rows).to_csv(root / 'dr' / 'metadata.csv', index=False)
    training = TrainingConfig(batch_size=2, image_size=8, pretrained=False, epochs=3)
    config = experiment.ExperimentConfig(training=training, training_mode='fast', seed=13,
                                         budget=4, device='cpu', data_root=root, output_dir=tmp_path / 'results')
    def factory(num_classes, pretrained, device):
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes)).to(device)
    result = experiment.run_experiment(config, model_factory=factory)
    assert len(result.curves['method'].unique()) == 6
    assert set(result.curves.risk_objective) == {'unweighted_cross_entropy', 'class_balanced_cross_entropy'}
    assert result.curves.groupby(['method', 'risk_objective']).n_labels.max().eq(4).all()
    assert result.training_metadata['mode'] == 'fast'
    assert result.reference_metrics['loss'] >= 0
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata['configuration']['seed'] == 13
    assert metadata['training']['mode'] == 'fast'
    assert metadata['class_counts'] == [10, 10]
    assert metadata['evaluation_size'] == 4
    saved = pd.read_csv(result.curves_path)
    pd.testing.assert_frame_equal(saved, result.curves, check_dtype=False)
    # Repeated seeded runs have the same estimates and cannot overwrite artifacts.
    repeated = experiment.run_experiment(config, model_factory=factory)
    assert result.curves_path != repeated.curves_path
    pd.testing.assert_frame_equal(result.curves, repeated.curves)


def test_cli_selects_fast_training_and_independent_objectives(monkeypatch):
    received = []
    monkeypatch.setattr(experiment, 'run_experiment', lambda config: received.append(config))
    experiment.main(['--training-mode', 'fast', '--no-weighted-loss', '--selection-objective',
                     'class_balanced_cross_entropy', '--no-class-balanced-at', '--seed', '7'])
    assert received[0].training_mode == 'fast'
    assert received[0].training.weighted_loss is False
    assert received[0].class_balanced_at is False
    assert received[0].seed == 7
