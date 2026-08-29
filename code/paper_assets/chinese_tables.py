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

import re

from paper_assets.paths import DATA

DATA_DIR = DATA
CHINESE_DIR = DATA / "zh"

#: (file name, Chinese caption, Chinese header row).  The captions follow the
#: English ones sentence for sentence; the headers name the same columns.
TABLES = [
    (
        "experiment1_rewrite_summary_table.tex",
        "精确二重特征值 \\(\\lambda_7=\\lambda_8\\) 的经验误差统计，"
        "基于每个 \\(N\\) 上 100 次独立特征抽样。",
        "$N$ & $Q_{0.9}(\\zeta_N(F_\\star))$ & "
        "$Q_{0.9}(\\operatorname{gap}_{H^1})$ & $Q_{0.9}(e_{\\rm cl})$\\\\",
    ),
    (
        "experiment2_rewrite_summary_table.tex",
        "径向梯度单位球前十个特征值的最大相对误差与计算时间，"
        "分别对应等参 \\(P_2\\) 有限元方法、Eig-PIELM 与 RFM。"
        "RFM 各行为二十次独立特征抽样的中位数，方括号内为观测范围。",
        "方法 & 分辨率 & 时间 (s) & 最大相对误差\\\\",
    ),
    (
        "experiment3_rewrite_summary_table.tex",
        "十维基准前三个有序特征值的相对误差。"
        "RFM 各行为二十次独立特征抽样的中位数。",
        "势函数 & 特征值 & DRM 误差 & RFM 误差\\\\",
    ),
    (
        "experiment4_rewrite_summary_table.tex",
        "IAEA 四分之一堆芯基准上神经网络基线与 RFM 的误差和计算时间。"
        "RFM 各行为二十次独立特征抽样的中位数，方括号内为观测范围。",
        "方法 & $k_{\\rm eff}$ & $e_k$ & $e_\\phi$ & 时间 (s)\\\\",
    ),
    (
        "experiment5_nonlinear_gpe_table.tex",
        "平移 Gross--Pitaevskii 问题上弱 Galerkin 格式与 RFM 的误差和计算"
        "时间。RFM 各行为五次独立特征抽样的中位数，每次抽样计时五遍；"
        "误差范围取自各次抽样，时间范围取自各次计时调用。",
        "方法 & 分辨率 & 时间 (s) & \\(\\lvert\\lambda-\\lambda_{\\rm ref}\\rvert\\) & "
        "\\(\\lvert E-E_{\\rm ref}\\rvert\\)\\\\",
    ),
    (
        "experiment6_dipolar_bec_table.tex",
        "旋转双组分偶极凝聚体上 GFLM--KTM 与 RFM 的误差和计算时间。"
        "RFM 各行为十次独立特征抽样的中位数，GFLM--KTM 各行为二十个已发表"
        "初值的中位数，方括号内为观测范围。",
        "方法 & 分辨率 & 时间 (s) & \\(\\lvert E-E_{\\rm ref}\\rvert\\) & "
        "\\(\\lvert\\mu_1-\\mu_{1,\\rm ref}\\rvert\\) & "
        "\\(\\lvert\\mu_2-\\mu_{2,\\rm ref}\\rvert\\) & \\(\\lvert I_g\\rvert\\)\\\\",
    ),
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
    for name, caption, header in TABLES:
        english = (DATA_DIR / name).read_text(encoding="utf-8")
        (CHINESE_DIR / name).write_text(
            translate(english, caption, header).rstrip() + "\n", encoding="utf-8"
        )
        print(f"  wrote {CHINESE_DIR.name}/{name}")


def main() -> None:
    build_chinese_tables()


if __name__ == "__main__":
    main()
