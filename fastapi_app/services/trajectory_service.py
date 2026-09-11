# trajectory_service.py
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from models.domain_models import ChildProfile, ScreeningAssessment, AssessmentRecommendation
from schemas.domain_schemas import TrajectoryResponse, TrajectoryTimelineItem, MilestoneShiftItem
from core import QUESTION_LABELS


def is_flagged(code: str, score: int) -> bool:
    """Returns True if Q-CHAT ordinal score (0-4) meets clinical delay threshold (>=3)."""
    return score >= 3


class TrajectoryService:
    @staticmethod
    def get_trajectory(db: Session, child: ChildProfile) -> TrajectoryResponse:
        assessments: List[ScreeningAssessment] = (
            db.query(ScreeningAssessment)
            .filter(ScreeningAssessment.child_id == child.id)
            .order_by(ScreeningAssessment.completed_at.asc())
            .all()
        )

        total_assessments = len(assessments)
        if total_assessments == 0:
            return TrajectoryResponse(
                child_id=child.id,
                child_name=child.first_name,
                total_assessments=0,
                trend_direction="insufficient_data",
                baseline_risk=None,
                current_risk=None,
                risk_delta=None,
                assessments_timeline=[],
                milestone_shifts=[],
                summary_narrative=f"No screening assessments have been recorded yet for {child.first_name}."
            )

        timeline: List[TrajectoryTimelineItem] = []
        for a in assessments:
            # Determine flagged milestones
            scores = {
                "A1": a.a1_score, "A2": a.a2_score, "A3": a.a3_score, "A4": a.a4_score, "A5": a.a5_score,
                "A6": a.a6_score, "A7": a.a7_score, "A8": a.a8_score, "A9": a.a9_score, "A10": a.a10_score
            }
            flagged = [
                f"{k}: {QUESTION_LABELS[k]}"
                for k, v in scores.items()
                if is_flagged(k, v)
            ]

            # Find top app and book recommendations
            top_app = None
            top_book = None
            for rec in a.recommendations:
                if rec.resource_type == "app" and top_app is None:
                    top_app = rec.item_name
                elif rec.resource_type == "book" and top_book is None:
                    top_book = rec.item_name

            timeline.append(TrajectoryTimelineItem(
                assessment_id=a.id,
                completed_at=a.completed_at,
                age_months=a.age_months,
                risk_probability=a.risk_probability,
                is_high_risk=a.is_high_risk,
                total_flags=a.total_flags,
                flagged_milestones=flagged,
                top_recommended_app=top_app,
                top_recommended_book=top_book,
                profile_explained=a.profile_explained
            ))

        baseline_risk = assessments[0].risk_probability
        current_risk = assessments[-1].risk_probability

        if total_assessments == 1:
            risk_delta = 0.0
            trend_direction = "stable"
            latest = assessments[0]
            scores_curr = {
                "A1": latest.a1_score, "A2": latest.a2_score, "A3": latest.a3_score, "A4": latest.a4_score,
                "A5": latest.a5_score, "A6": latest.a6_score, "A7": latest.a7_score, "A8": latest.a8_score,
                "A9": latest.a9_score, "A10": latest.a10_score
            }
            milestone_shifts = [
                MilestoneShiftItem(
                    code=k,
                    label=QUESTION_LABELS[k],
                    previous_score=None,
                    current_score=v,
                    previous_flagged=None,
                    current_flagged=is_flagged(k, v),
                    status="Ongoing Focus Area" if is_flagged(k, v) else "Stable Typical"
                )
                for k, v in scores_curr.items()
            ]
            summary_narrative = (
                f"{child.first_name} has completed 1 baseline assessment with a risk probability "
                f"of {current_risk:.1f}% ({latest.total_flags} milestone flags). "
                f"A follow-up screening in 30 days will establish a developmental trajectory."
            )
        else:
            latest = assessments[-1]
            previous = assessments[-2]
            risk_delta = round(current_risk - previous.risk_probability, 2)

            if risk_delta <= -3.0:
                trend_direction = "improving"
            elif risk_delta >= 3.0:
                trend_direction = "concerning"
            else:
                trend_direction = "stable"

            scores_prev = {
                "A1": previous.a1_score, "A2": previous.a2_score, "A3": previous.a3_score, "A4": previous.a4_score,
                "A5": previous.a5_score, "A6": previous.a6_score, "A7": previous.a7_score, "A8": previous.a8_score,
                "A9": previous.a9_score, "A10": previous.a10_score
            }
            scores_curr = {
                "A1": latest.a1_score, "A2": latest.a2_score, "A3": latest.a3_score, "A4": latest.a4_score,
                "A5": latest.a5_score, "A6": latest.a6_score, "A7": latest.a7_score, "A8": latest.a8_score,
                "A9": latest.a9_score, "A10": latest.a10_score
            }

            milestone_shifts = []
            resolved_count = 0
            new_flag_count = 0

            for k in QUESTION_LABELS:
                prev_score = scores_prev.get(k, 0)
                curr_score = scores_curr.get(k, 0)
                prev_flag = is_flagged(k, prev_score)
                curr_flag = is_flagged(k, curr_score)

                if prev_flag and not curr_flag:
                    status = "Resolved"
                    resolved_count += 1
                elif prev_flag and curr_flag:
                    status = "Ongoing Focus Area"
                elif not prev_flag and curr_flag:
                    status = "New Flag Identified"
                    new_flag_count += 1
                else:
                    status = "Stable Typical"

                milestone_shifts.append(MilestoneShiftItem(
                    code=k,
                    label=QUESTION_LABELS[k],
                    previous_score=prev_score,
                    current_score=curr_score,
                    previous_flagged=prev_flag,
                    current_flagged=curr_flag,
                    status=status
                ))

            if trend_direction == "improving":
                narrative_trend = f"shows positive developmental progress with a {abs(risk_delta):.1f}% reduction in ASD risk probability."
            elif trend_direction == "concerning":
                narrative_trend = f"shows an elevated risk delta (+{risk_delta:.1f}%), suggesting focused clinical consultation is warranted."
            else:
                narrative_trend = f"demonstrates stable milestone stability across the last evaluation period."

            summary_narrative = (
                f"{child.first_name} {narrative_trend} "
                f"Currently, {resolved_count} previously flagged milestone(s) have resolved, "
                f"while {latest.total_flags} area(s) remain ongoing points of developmental support."
            )

        return TrajectoryResponse(
            child_id=child.id,
            child_name=child.first_name,
            total_assessments=total_assessments,
            trend_direction=trend_direction,
            baseline_risk=round(baseline_risk, 1),
            current_risk=round(current_risk, 1),
            risk_delta=risk_delta,
            assessments_timeline=timeline,
            milestone_shifts=milestone_shifts,
            summary_narrative=summary_narrative
        )
