"""The Chinese tables, derived from the English ones rather than kept by hand.

The Chinese manuscript is a translation of the English one, so its tables have
to carry the same numbers.  Maintaining two sets by hand is what let them drift
two generations apart: the Chinese Table 2 still described a tetrahedron with a
column of degrees of freedom the English had already dropped.

Only two things differ between a table and its translation -- the caption and
the header row -- because every cell below the header is a number or a LaTeX
control sequence.  So the English table is generated first, and this module
rewrites those two lines.  A number can then only be wrong in both languages at
once, which is the property worth having.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from paper_assets.paths import DATA, REPO_DATA
from paper_assets.records import draw_counts, read_converged, repetitions_per_draw

DATA_DIR = DATA
CHINESE_DIR = DATA / "zh"

def table_specs():
    """Translate the current statistics and sample counts, preserving numeric cells."""
    exp1 = pd.read_csv(REPO_DATA / "experiment1_theory_matched_trials.csv")
    exp2 = json.loads((REPO_DATA / "experiment2_graded_ball_run.json").read_text())
    exp4 = pd.read_csv(REPO_DATA / "runs/experiment4/rfm_rows.csv")
    exp5 = read_converged(REPO_DATA / "experiment5_rfm_timed.csv")
    exp6 = read_converged(REPO_DATA / "experiment6_rfm_timed.csv")
    counts = draw_counts(exp5)
    budgets = ",".join(str(n) for n in counts)
    sizes = ", ".join(str(n) for n in counts.values())
    n1 = int(exp1.groupby("N").size().iloc[0])
    n2 = next(row["draws"] for row in exp2["rows"] if row["method"] == "RFM")
    n4 = int(exp4.seed.nunique())
    n6 = next(iter(draw_counts(exp6).values()))
    return [
        ("experiment1_rewrite_summary_table.tex",
         rf"精确二重特征值 \(\lambda_7=\lambda_8\) 的误差。每项为 {n1} 次独立特征抽样的算术均值。",
         r"$N$ & $\zeta_N(F_\star)$ & $d_{H^1}(E_\star,E_{\star,N})$ & $e_{\rm cl}$\\"),
        ("experiment2_rewrite_summary_table.tex",
         rf"径向梯度单位球前十个特征值的最大相对误差与计算时间，分别对应等参 \(P_2\) 有限元方法、Eig-PIELM 与 RFM。有限元的 \(n\) 为网格加密次数。RFM 各行为 {n2} 次独立特征抽样的中位数。",
         r"方法 & 分辨率 & 时间 (s) & 最大相对误差\\"),
        ("experiment4_rewrite_summary_table.tex",
         f"IAEA 四分之一堆芯基准上神经网络基线与 RFM 的误差和计算时间。RFM 各行为 {n4} 次独立特征抽样的中位数。",
         r"方法 & $k_{\rm eff}$ & $e_k$ & $e_\varphi$ & 时间 (s)\\"),
        ("experiment5_nonlinear_gpe_table.tex",
         rf"平移 Gross--Pitaevskii 问题上弱 Galerkin 格式与 RFM 的误差和计算时间。RFM 各行为收敛样本的中位数；\(N={budgets}\) 对应的样本数依次为 {sizes}，每次抽样计时 {repetitions_per_draw(exp5)} 遍。",
         r"方法 & 分辨率 & 时间 (s) & \(\lvert\lambda-\lambda_{\rm ref}\rvert\) & \(\lvert E-E_{\rm ref}\rvert\)\\"),
        ("experiment6_dipolar_bec_table.tex",
         f"旋转双组分偶极凝聚体上 GFLM--KTM 与 RFM 的误差和计算时间。RFM 各行为 {n6} 次独立特征抽样的中位数，GFLM--KTM 各行为从十种已发表初值的有序组合中选取的二十个初始状态的中位数。",
         r"方法 & 分辨率 & 时间 (s) & \(\lvert E-E_{\rm ref}\rvert\) & \(\lvert\mu_1-\mu_{1,\rm ref}\rvert\) & \(\lvert\mu_2-\mu_{2,\rm ref}\rvert\) & \(\lvert I_g\rvert\)\\"),
    ]


#: ``\caption{...}`` spans lines and contains braces, so it is matched from the
#: opening brace to the line before ``\label``, which every one of these tables
#: carries immediately after it.
_CAPTION = re.compile(r"\\caption\{.*?\}\n(?=\\label\{)", re.DOTALL)


def translate(english: str, caption: str, header: str) -> str:
    """Swap the caption and the header row; leave every number alone."""
    if not _CAPTION.search(english):
        raise ValueError("no caption followed by a label")
    # a lambda, because re.sub reads backslashes in a replacement string and
    # every one of these captions is full of them
    chinese = _CAPTION.sub(
        lambda _match: "\\caption{" + caption + "}\n", english, count=1
    )

    lines = chinese.split("\n")
    try:
        rule = lines.index("\\toprule")
    except ValueError as error:  # pragma: no cover - the tables all have one
        raise ValueError("no \\toprule") from error
    lines[rule + 1] = header
    return "\n".join(lines)


def build_chinese_tables() -> None:
    """Regenerate every Chinese table from the English one beside it."""
    CHINESE_DIR.mkdir(parents=True, exist_ok=True)
    for name, caption, header in table_specs():
        source = DATA_DIR / name
        english = (source if source.exists() else REPO_DATA / name).read_text(encoding="utf-8")
        (CHINESE_DIR / name).write_text(
            translate(english, caption, header).rstrip() + "\n", encoding="utf-8", newline="\n"
        )
        print(f"  wrote {CHINESE_DIR.name}/{name}")


def main() -> None:
    build_chinese_tables()


if __name__ == "__main__":
    main()
