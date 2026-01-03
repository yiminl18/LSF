"""Provenance node annotation using judge_header."""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from src.labeling.judge import judge_header


class ProvenanceAnnotator:
    """Annotate candidate headers as provenance/non-provenance nodes."""
    
    def __init__(
        self,
        min_entries: int = 10,
        hard_cap: int = 20,
        stop_after_positive: bool = True,
        gpt_key_path: str = None,
    ):
        """
        Initialize annotator.
        
        Args:
            min_entries: Minimum entries to annotate per PDF
            hard_cap: Maximum entries if no positive found
            stop_after_positive: Stop after finding positive if >= min_entries
            gpt_key_path: Path to GPT API key
        """
        self.min_entries = min_entries
        self.hard_cap = hard_cap
        self.stop_after_positive = stop_after_positive
        self.gpt_key_path = gpt_key_path
    
    def annotate_pdf(
        self,
        candidates: List[Dict[str, Any]],
        question: str,
        ground_truth: Any,
    ) -> List[Dict[str, Any]]:
        """
        Annotate candidates for a single PDF + question.
        
        Args:
            candidates: List of candidate headers (sorted by baseline similarity)
                Each candidate should have: text, features (bbox, font, etc.)
            question: Question text
            ground_truth: Ground truth answer
            
        Returns:
            List of annotated entries with label (1=prov, 0=non-prov)
        """
        entries = []
        positives = 0
        
        for rank_idx, cand in enumerate(candidates, start=1):
            # Call judge_header for labeling
            try:
                matched, resp = judge_header(
                    text=cand['text'],
                    question=question,
                    ground_truth=ground_truth,
                    key_path=self.gpt_key_path or "",
                )
            except Exception as e:
                print(f"[judge_error] {e} -> treat as not matched")
                matched = False
            
            # Build entry
            entry = {
                'text': cand['text'],
                'label': 1 if matched else 0,
                'rank_index': rank_idx,
                **cand.get('features', {}),
            }
            entries.append(entry)
            
            if matched:
                positives += 1
            
            # Stopping policy
            if len(entries) >= self.min_entries and positives >= 1:
                if self.stop_after_positive:
                    break
            
            if len(entries) >= min(len(candidates), self.hard_cap):
                break
        
        return entries
    
    def annotate_dataset(
        self,
        pdf_ids: List[str],
        question_indices: List[int],
        questions: List[str],
        merged_dir: Path,
        ground_truth_dir: Path,
        embedding_dir: Path,
        output_path: Path,
        skip_existing: bool = True,
    ) -> Dict[str, int]:
        """
        Annotate multiple PDFs for multiple questions.
        
        Args:
            pdf_ids: List of PDF IDs to process
            question_indices: List of question indices (1-based)
            questions: Full question list
            merged_dir: Directory with *_merged.json files
            ground_truth_dir: Directory with ground truth
            embedding_dir: Directory with embeddings
            output_path: Output JSONL path
            skip_existing: Skip (pdf, question) pairs already in output
            
        Returns:
            Stats dict with counts
        """
        # Load existing entries if skip_existing
        existing_pairs = set()
        if skip_existing and output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        existing_pairs.add((obj['pdf_id'], obj['question_index']))
        
        stats = {
            'total_pairs': 0,
            'skipped': len(existing_pairs),
            'processed': 0,
            'errors': 0,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'a', encoding='utf-8') as f_out:
            for pdf_id in pdf_ids:
                for q_idx in question_indices:
                    # Skip if exists
                    if (pdf_id, q_idx) in existing_pairs:
                        continue
                    
                    stats['total_pairs'] += 1
                    
                    # Check ground truth is answerable
                    gt = self._load_ground_truth(ground_truth_dir, pdf_id, q_idx)
                    if not self._is_answerable(gt):
                        print(f"[skip] {pdf_id} q{q_idx}: not answerable")
                        continue
                    
                    # Get question text
                    if q_idx < 1 or q_idx > len(questions):
                        print(f"[skip] {pdf_id} q{q_idx}: invalid question index")
                        continue
                    question = questions[q_idx - 1]
                    
                    # Load candidates (requires baseline ranking logic)
                    # This would call into src/evaluation/ or similar
                    # For now, stub this - in practice, reuse your existing logic
                    print(f"[process] {pdf_id} q{q_idx}")
                    
                    # TODO: Implement candidate loading + ranking
                    # candidates = self._get_candidates_sorted(pdf_id, question, ...)
                    # entries = self.annotate_pdf(candidates, question, gt)
                    
                    # for entry in entries:
                    #     record = {
                    #         'pdf_id': pdf_id,
                    #         'question_index': q_idx,
                    #         'question_text': question,
                    #         **entry,
                    #     }
                    #     f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
                    stats['processed'] += 1
        
        return stats
    
    def _load_ground_truth(self, gt_dir: Path, pdf_id: str, q_idx: int) -> Any:
        """Load ground truth for (pdf, question)."""
        p = gt_dir / f"{pdf_id}.txt_answers.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding='utf-8'))
        return data.get(str(q_idx))
    
    def _is_answerable(self, gt_value: Any) -> bool:
        """Check if ground truth is answerable."""
        if gt_value is None:
            return False
        if isinstance(gt_value, (list, dict)):
            return bool(gt_value)
        s = str(gt_value).strip()
        if not s:
            return False
        s_norm = s.lower()
        if s_norm in {"none", "null", "n/a", "na", "not applicable"}:
            return False
        return True


