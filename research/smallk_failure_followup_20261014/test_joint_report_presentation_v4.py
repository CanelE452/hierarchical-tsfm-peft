"""Two presentation fixtures: no model/checkpoint/scientific array reads."""
import ast
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def test_numerical_adapters_preserved_across_presentation_derivative():
    def functions(name):
        parsed=ast.parse((HERE/name).read_text(encoding='utf-8'))
        return {n.name:ast.dump(n,include_attributes=False) for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    previous=functions('make_joint_followup_report_v3.py');current=functions('make_joint_followup_report_v4.py')
    assert all(node==current[name] for name,node in previous.items() if name not in ('figure_learning','generate'))


def test_log_labels_ASCII_and_selected_annotations_inside_axes(monkeypatch,tmp_path):
    from research.smallk_failure_followup_20261014 import make_joint_followup_report_v4 as report
    plt,font=report.old.configure_plotting();captured=[]
    monkeypatch.setattr(report.old,'finish_figure',lambda fig,*args,**kwargs:captured.append(fig))
    learning=[];gradients=[]
    for dataset in report.DATASETS:
        for seed in report.SEEDS:
            for index,objective in enumerate(report.OBJECTIVES):
                learning.append({'dataset':dataset,'seed':seed,'objective':objective,'E_changed_tensors':2,'D_changed_tensors':3,'selected_new_update':100})
                for update in (0,99,199,399,599):
                    gradients.append({'dataset':dataset,'seed':seed,'objective':objective,'new_update':update,'F_input_gradient_norm':float(10.**(-4+index))*(1+update/600)})
    report.figure_learning({'joint_learning':learning,'joint_gradients':gradients},tmp_path,plt)
    figure=captured[0];figure.canvas.draw()
    for axis in (figure.axes[1],figure.axes[3]):
        lower,upper=axis.get_ylim()
        assert all(lower<text.get_position()[1]<upper for text in axis.texts)
        assert all('\u2212' not in label.get_text() and label.get_text().isascii() for label in axis.get_yticklabels())
    plt.close(figure)
