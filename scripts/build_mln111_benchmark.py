"""Build the reviewable 30-question MLN111 benchmark from exact page blocks."""

# ruff: noqa: E501 - benchmark answers are kept as readable review units.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from viettheory.benchmark import BenchmarkQuestion, BenchmarkSplit, QuestionType
from viettheory.ids import stable_id
from viettheory.schema import Page, SourceSpan


@dataclass(frozen=True)
class Spec:
    question: str
    answer: str
    kind: QuestionType
    split: BenchmarkSplit
    difficulty: str
    evidence: tuple[tuple[int, str], ...] = ()
    answerable: bool = True


def _span(page: Page, needle: str) -> SourceSpan:
    for block in page.blocks:
        if needle.casefold() in block.text.casefold():
            return SourceSpan(
                page_id=page.page_id,
                pdf_page=page.pdf_page,
                printed_page=page.printed_page,
                bbox=block.bbox,
                text=block.text,
            )
    raise ValueError(f"Evidence not found on PDF page {page.pdf_page}: {needle!r}")


def _specs() -> tuple[Spec, ...]:
    d = BenchmarkSplit.DEVELOPMENT
    t = BenchmarkSplit.TEST
    definition = QuestionType.DEFINITION
    explanation = QuestionType.EXPLANATION
    paraphrase = QuestionType.PARAPHRASE
    multi = QuestionType.MULTI_HOP
    false = QuestionType.FALSE_PREMISE
    ood = QuestionType.OUT_OF_DOMAIN
    return (
        Spec(
            "Vật chất được V.I. Lênin định nghĩa như thế nào?",
            "Vật chất là phạm trù triết học chỉ thực tại khách quan, được cảm giác phản ánh và tồn tại không lệ thuộc vào cảm giác.",
            definition,
            d,
            "easy",
            ((73, "Vật chất là một phạm trù triết học"),),
        ),
        Spec(
            "Ý thức là gì theo quan điểm duy vật biện chứng?",
            "Ý thức là sự phản ánh thế giới hiện thực bởi bộ óc con người và là hình thức phản ánh cao nhất của thế giới vật chất.",
            definition,
            d,
            "easy",
            ((88, "Ý thức là sự phản ánh"),),
        ),
        Spec(
            "Thực tiễn là gì?",
            "Thực tiễn là toàn bộ hoạt động vật chất-cảm tính, có tính lịch sử-xã hội của con người nhằm cải tạo tự nhiên và xã hội.",
            definition,
            d,
            "easy",
            ((151, "thực tiễn là toàn bộ"),),
        ),
        Spec(
            "Phủ định biện chứng là gì?",
            "Phủ định biện chứng là sự phủ định tạo tiền đề và điều kiện cho phát triển, nối sự vật cũ với sự vật mới.",
            definition,
            d,
            "easy",
            ((142, "Phủ định biện chứng là khái niệm"),),
        ),
        Spec(
            "Phương thức sản xuất là gì?",
            "Phương thức sản xuất là sự thống nhất giữa lực lượng sản xuất ở một trình độ nhất định và quan hệ sản xuất tương ứng.",
            definition,
            d,
            "easy",
            ((166, "Phương thức sản xuất là sự thống nhất"),),
        ),
        Spec(
            "Lực lượng sản xuất là gì?",
            "Lực lượng sản xuất là sự kết hợp giữa người lao động với tư liệu sản xuất, tạo sức sản xuất và năng lực cải biến tự nhiên.",
            definition,
            d,
            "easy",
            ((166, "Lực lượng sản xuất là sự kết hợp"),),
        ),
        Spec(
            "Giai cấp được V.I. Lênin định nghĩa như thế nào?",
            "Giai cấp là những tập đoàn người khác nhau về địa vị trong hệ thống sản xuất xã hội; một tập đoàn có thể chiếm đoạt lao động của tập đoàn khác.",
            definition,
            t,
            "medium",
            ((189, "Giai cấp là những tập đoàn người"),),
        ),
        Spec(
            "Tồn tại xã hội là gì?",
            "Tồn tại xã hội là toàn bộ sinh hoạt vật chất và các điều kiện sinh hoạt vật chất của xã hội.",
            definition,
            t,
            "easy",
            ((240, "Tồn tại xã hội là toàn bộ"),),
        ),
        Spec(
            "Nhà nước là gì xét về quyền lực chính trị?",
            "Nhà nước là tổ chức đặc biệt của quyền lực chính trị, dựa trên hệ tư tưởng và các hình thức kiểm soát xã hội.",
            definition,
            t,
            "medium",
            ((178, "Nhà nước là tổ chức đặc biệt"),),
        ),
        Spec(
            "Triết học xuất hiện với tư cách loại hình tri thức nào?",
            "Triết học là dạng tri thức lý luận xuất hiện sớm nhất và là một hình thái ý thức xã hội có nguồn gốc nhận thức và xã hội.",
            definition,
            t,
            "medium",
            ((7, "Triết học là dạng tri thức lý luận"),),
        ),
        Spec(
            "Vì sao thực tiễn được xem là tiêu chuẩn của chân lý?",
            "Thực tiễn cung cấp tiêu chuẩn khách quan để kiểm nghiệm tri thức và chân lý trong hoạt động cải tạo thế giới.",
            explanation,
            d,
            "medium",
            ((48, "thực tiễn là tiêu chuẩn khách quan của chân lý"),),
        ),
        Spec(
            "Vì sao ý thức có nguồn gốc xã hội chứ không chỉ nguồn gốc tự nhiên?",
            "Bộ óc tạo tiền đề vật chất, nhưng hoạt động thực tiễn xã hội trực tiếp quyết định sự ra đời của ý thức; ý thức là sản phẩm xã hội.",
            explanation,
            d,
            "medium",
            ((88, "Hoạt động thực tiễn của loài người"),),
        ),
        Spec(
            "Giải thích vì sao phương thức sản xuất là sự thống nhất của hai loại quan hệ.",
            "Nó kết hợp quan hệ giữa con người với tự nhiên qua lực lượng sản xuất và quan hệ giữa người với người qua quan hệ sản xuất.",
            explanation,
            d,
            "medium",
            ((166, "hai mối quan hệ “song trùng”"),),
        ),
        Spec(
            "Vì sao tồn tại xã hội giữ vai trò quyết định đối với ý thức xã hội?",
            "Đời sống vật chất và phương thức sản xuất tạo cơ sở khách quan cho các quá trình xã hội, chính trị và tinh thần, nên ý thức phản ánh tồn tại xã hội.",
            explanation,
            t,
            "hard",
            ((240, "Phương thức sản xuất đời sống vật chất quyết định"),),
        ),
        Spec(
            "Tại sao mâu thuẫn nội tại được coi là nguồn gốc của vận động?",
            "Sự đấu tranh giữa các mặt đối lập trong bản thân sự vật tạo ra vận động, thay đổi và phát triển.",
            explanation,
            t,
            "medium",
            ((26, "Nguồn gốc của sự vận động"),),
        ),
        Spec(
            "Theo Lênin, cái gì tồn tại bên ngoài và không phụ thuộc cảm giác của con người?",
            "Đó là thực tại khách quan, nội dung mà phạm trù vật chất chỉ đến.",
            paraphrase,
            d,
            "medium",
            ((73, "tồn tại không lệ thuộc vào cảm giác"),),
        ),
        Spec(
            "Hoạt động nào mang tính vật chất-cảm tính và nhằm biến đổi tự nhiên, xã hội?",
            "Đó là thực tiễn.",
            paraphrase,
            d,
            "easy",
            ((151, "hoạt động vật chất - cảm tính"),),
        ),
        Spec(
            "Yếu tố nào kết hợp người lao động và tư liệu sản xuất?",
            "Đó là lực lượng sản xuất.",
            paraphrase,
            d,
            "easy",
            ((166, "Lực lượng sản xuất là sự kết hợp"),),
        ),
        Spec(
            "Khái niệm nào bao quát đời sống vật chất và điều kiện vật chất của xã hội?",
            "Đó là tồn tại xã hội.",
            paraphrase,
            t,
            "easy",
            ((240, "Tồn tại xã hội là toàn bộ"),),
        ),
        Spec(
            "Sự thay thế cái cũ mà vẫn tạo liên hệ và tiền đề phát triển cho cái mới gọi là gì?",
            "Đó là phủ định biện chứng.",
            paraphrase,
            t,
            "medium",
            ((143, "yếu tố liên hệ giữa sự vật"),),
        ),
        Spec(
            "Phân tích đồng thời nguồn gốc tự nhiên và xã hội của ý thức.",
            "Nguồn gốc tự nhiên là bộ óc người có năng lực phản ánh; nguồn gốc xã hội trực tiếp là hoạt động thực tiễn, lao động và đời sống xã hội.",
            multi,
            d,
            "hard",
            ((88, "óc của con người có năng lực"), (88, "nguồn gốc xã hội")),
        ),
        Spec(
            "Nêu quan hệ giữa lực lượng sản xuất, quan hệ sản xuất và phương thức sản xuất.",
            "Phương thức sản xuất là sự thống nhất giữa lực lượng sản xuất và quan hệ sản xuất; hai mặt biểu hiện quan hệ người-tự nhiên và người-người trong sản xuất.",
            multi,
            d,
            "hard",
            ((166, "Phương thức sản xuất là sự thống nhất"), (166, "hai mối quan hệ “song trùng”")),
        ),
        Spec(
            "Liên hệ tồn tại xã hội với ý thức xã hội và chỉ ra yếu tố vật chất cơ bản nhất.",
            "Ý thức xã hội phản ánh tồn tại xã hội; trong tồn tại xã hội, phương thức sản xuất vật chất là yếu tố cơ bản nhất.",
            multi,
            t,
            "hard",
            (
                (240, "ý thức xã hội phản ánh"),
                (240, "vật chất là yếu tố cơ bản nhất"),
            ),
        ),
        Spec(
            "Ý thức là thực thể bẩm sinh tồn tại độc lập với xã hội, đúng hay sai?",
            "Sai. Ý thức là sự phản ánh hiện thực bởi bộ óc người và là sản phẩm xã hội hình thành qua hoạt động thực tiễn.",
            false,
            d,
            "medium",
            ((88, "ý thức đã là một sản phẩm xã hội"),),
            False,
        ),
        Spec(
            "Phủ định biện chứng xóa bỏ sạch mọi yếu tố của cái cũ, đúng hay sai?",
            "Sai. Phủ định biện chứng có tính kế thừa và tạo liên hệ giữa cái cũ với cái mới.",
            false,
            d,
            "medium",
            ((143, "tính kế thừa"),),
            False,
        ),
        Spec(
            "Phương thức sản xuất chỉ là quan hệ giữa người với tự nhiên, đúng hay sai?",
            "Sai. Nó đồng thời bao gồm quan hệ người-tự nhiên và quan hệ giữa người với người trong sản xuất.",
            false,
            t,
            "medium",
            ((166, "hai mối quan hệ “song trùng”"),),
            False,
        ),
        Spec(
            "Thời tiết Hà Nội ngày mai thế nào?",
            "Câu hỏi nằm ngoài phạm vi giáo trình MLN111.",
            ood,
            d,
            "easy",
            (),
            False,
        ),
        Spec(
            "Viết một hàm Python sắp xếp danh sách.",
            "Câu hỏi nằm ngoài phạm vi giáo trình MLN111.",
            ood,
            d,
            "easy",
            (),
            False,
        ),
        Spec(
            "Tính đạo hàm của sin(x).",
            "Câu hỏi nằm ngoài phạm vi giáo trình MLN111.",
            ood,
            t,
            "easy",
            (),
            False,
        ),
        Spec(
            "Cấu trúc ADN gồm những thành phần nào?",
            "Câu hỏi nằm ngoài phạm vi giáo trình MLN111.",
            ood,
            t,
            "easy",
            (),
            False,
        ),
    )


def main() -> int:
    pages_path = Path("data/processed/MLN111/pages.jsonl")
    output_path = Path("benchmark/mln111_questions.jsonl")
    pages = {
        page.pdf_page: page
        for line in pages_path.read_text(encoding="utf-8").splitlines()
        if (page := Page.model_validate_json(line))
    }
    questions: list[BenchmarkQuestion] = []
    for index, spec in enumerate(_specs(), start=1):
        evidence = tuple(_span(pages[page], needle) for page, needle in spec.evidence)
        questions.append(
            BenchmarkQuestion(
                question_id=stable_id("question", "MLN111", index, spec.question),
                question=spec.question,
                subject_code="MLN111",
                question_type=spec.kind,
                split=spec.split,
                answerable=spec.answerable,
                difficulty=spec.difficulty,
                gold_answer=spec.answer,
                gold_evidence=evidence,
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for question in questions:
            output.write(json.dumps(question.model_dump(mode="json"), ensure_ascii=False) + "\n")
    review_path = output_path.with_name("mln111_review.md")
    review_lines = ["# MLN111 Benchmark Review", ""]
    for index, question in enumerate(questions, start=1):
        pages_label = (
            ", ".join(
                f"PDF {span.pdf_page} / printed {span.printed_page or '?'}"
                for span in question.gold_evidence
            )
            or "none (out of domain)"
        )
        evidence_preview = " ".join(span.text for span in question.gold_evidence)[:500]
        review_lines.extend(
            [
                f"## {index}. {question.question}",
                "",
                f"- Type: `{question.question_type.value}`",
                f"- Split: `{question.split.value}`",
                f"- Answerable: `{question.answerable}`",
                f"- Gold pages: {pages_label}",
                f"- Gold answer: {question.gold_answer}",
                f"- Evidence: {evidence_preview or 'N/A'}",
                "- Human decision: `[ ] approve` `[ ] revise` `[ ] reject`",
                "",
            ]
        )
    review_path.write_text("\n".join(review_lines), encoding="utf-8")
    print(f"Wrote {len(questions)} draft questions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
