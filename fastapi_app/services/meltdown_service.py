# meltdown_service.py
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from uuid import UUID
from collections import Counter
from sqlalchemy.orm import Session
from models.domain_models import ChildProfile, MeltdownIncident, MeltdownStrategyApplied
from schemas.domain_schemas import MeltdownTriageCreate, MeltdownAnalyticsResponse


DEFAULT_CALMING_STRATEGIES = [
    {
        "name": "Deep Pressure / Firm Hug",
        "category": "proprioceptive",
        "icon": "🤗",
        "action": "Apply firm, steady hug, weighted lap pad, or gentle compression along arms.",
        "rationale": "Activates the parasympathetic nervous system to down-regulate acute sensory overload."
    },
    {
        "name": "Noise Reduction / Mute",
        "category": "auditory",
        "icon": "🎧",
        "action": "Equip noise-cancelling headphones or move to a whisper-quiet space.",
        "rationale": "Cuts auditory input overload immediately."
    },
    {
        "name": "Dim Lights & Visual Reduction",
        "category": "visual",
        "icon": "💡",
        "action": "Lower blinds, turn off fluorescent lights, and face away from screens/windows.",
        "rationale": "Reduces visual hyper-stimulation."
    },
    {
        "name": "4-7-8 Guided Breathing",
        "category": "breathing",
        "icon": "🫁",
        "action": "Guide slow synchronized breathing: 4s inhale, 7s hold, 8s gentle exhale.",
        "rationale": "Regulates vagal tone and slows racing heart rate."
    },
    {
        "name": "Heavy Work & Push",
        "category": "proprioceptive",
        "icon": "🧱",
        "action": "Offer wall push-ups, squeezing a sensory putty, or carrying a weighted pillow.",
        "rationale": "Channels intense motor energy into stabilizing proprioceptive grounding."
    },
    {
        "name": "Cool Sensory Reset",
        "category": "tactile",
        "icon": "🧊",
        "action": "Offer a cool damp washcloth on cheeks/wrists or a sip of cold water.",
        "rationale": "Mild cold stimulation activates the mammalian dive reflex to curb panic."
    },
]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class MeltdownService:
    @staticmethod
    def get_quick_calming_strategies(db: Session, child_id: UUID) -> List[Dict[str, Any]]:
        return DEFAULT_CALMING_STRATEGIES

    @staticmethod
    def log_meltdown_triage(
        db: Session,
        child_id: UUID,
        triage_in: MeltdownTriageCreate
    ) -> MeltdownIncident:
        start = triage_in.start_time or datetime.now(timezone.utc)
        end = triage_in.end_time or datetime.now(timezone.utc)
        duration = triage_in.duration_seconds
        if duration <= 0 and triage_in.end_time and triage_in.start_time:
            duration = max(0, int((triage_in.end_time - triage_in.start_time).total_seconds()))

        incident = MeltdownIncident(
            child_id=child_id,
            start_time=start,
            end_time=end,
            duration_seconds=duration,
            intensity=triage_in.intensity,
            location=triage_in.location.lower().strip() if triage_in.location else "home",
            triggers=[t.strip().lower() for t in triage_in.triggers if t.strip()],
            notes=triage_in.notes
        )
        db.add(incident)
        db.flush()

        for s in triage_in.strategies:
            strategy_record = MeltdownStrategyApplied(
                meltdown_id=incident.id,
                strategy_name=s.strategy_name.strip(),
                strategy_category=s.strategy_category.strip().lower(),
                efficacy=s.efficacy.strip().lower(),
                notes=s.notes
            )
            db.add(strategy_record)

        db.commit()
        db.refresh(incident)
        return incident

    @staticmethod
    def get_child_incidents(db: Session, child_id: UUID, limit: int = 50) -> List[MeltdownIncident]:
        return (
            db.query(MeltdownIncident)
            .filter(MeltdownIncident.child_id == child_id)
            .order_by(MeltdownIncident.start_time.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def compute_environmental_patterns(db: Session, child: ChildProfile) -> MeltdownAnalyticsResponse:
        incidents = (
            db.query(MeltdownIncident)
            .filter(MeltdownIncident.child_id == child.id)
            .order_by(MeltdownIncident.start_time.asc())
            .all()
        )

        total_incidents = len(incidents)
        if total_incidents == 0:
            return MeltdownAnalyticsResponse(
                child_id=child.id,
                child_name=child.first_name,
                total_incidents=0,
                avg_duration_minutes=0.0,
                avg_intensity=0.0,
                trigger_frequencies={},
                location_frequencies={},
                time_of_day_distribution={"morning": 0, "afternoon": 0, "evening": 0, "night": 0},
                day_of_week_distribution={d: 0 for d in DAY_NAMES},
                strategy_efficacy_ranking=[],
                plain_english_insights=[
                    f"No sensory meltdown incidents have been logged yet for {child.first_name}.",
                    "Use the 1-Tap Active Meltdown screen during acute moments to capture duration and behavioral triggers."
                ]
            )

        # Durations & Intensity
        total_duration_sec = sum(i.duration_seconds for i in incidents)
        avg_duration_min = round(total_duration_sec / (total_incidents * 60.0), 1)
        avg_intensity = round(sum(i.intensity for i in incidents) / float(total_incidents), 1)

        # Trigger counts
        all_triggers: List[str] = []
        for i in incidents:
            if isinstance(i.triggers, list):
                all_triggers.extend([t.lower() for t in i.triggers])
        trigger_counts = dict(Counter(all_triggers).most_common())

        # Location counts
        location_counts = dict(Counter(i.location.lower() for i in incidents).most_common())

        # Time of day distribution
        time_dist = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
        for i in incidents:
            hour = i.start_time.hour
            if 6 <= hour < 12:
                time_dist["morning"] += 1
            elif 12 <= hour < 17:
                time_dist["afternoon"] += 1
            elif 17 <= hour < 21:
                time_dist["evening"] += 1
            else:
                time_dist["night"] += 1

        # Day of week distribution
        day_dist = {d: 0 for d in DAY_NAMES}
        for i in incidents:
            day_name = DAY_NAMES[i.start_time.weekday()]
            day_dist[day_name] += 1

        # Strategy Efficacy Rankings
        all_strategies: List[MeltdownStrategyApplied] = (
            db.query(MeltdownStrategyApplied)
            .join(MeltdownIncident, MeltdownStrategyApplied.meltdown_id == MeltdownIncident.id)
            .filter(MeltdownIncident.child_id == child.id)
            .all()
        )

        strat_stats: Dict[str, Dict[str, Any]] = {}
        for s in all_strategies:
            name = s.strategy_name
            if name not in strat_stats:
                strat_stats[name] = {
                    "strategy_name": name,
                    "category": s.strategy_category,
                    "times_used": 0,
                    "effective": 0,
                    "partially_effective": 0,
                    "ineffective": 0
                }
            strat_stats[name]["times_used"] += 1
            if s.efficacy == "effective":
                strat_stats[name]["effective"] += 1
            elif s.efficacy == "partially_effective":
                strat_stats[name]["partially_effective"] += 1
            else:
                strat_stats[name]["ineffective"] += 1

        ranking: List[Dict[str, Any]] = []
        for name, data in strat_stats.items():
            used = data["times_used"]
            eff = data["effective"]
            part = data["partially_effective"]
            score = round(((eff * 1.0) + (part * 0.5)) / used * 100.0, 1) if used > 0 else 0.0
            ranking.append({
                "strategy_name": name,
                "category": data["category"],
                "times_used": used,
                "effective_count": eff,
                "success_rate": score
            })

        ranking.sort(key=lambda x: (x["success_rate"], x["times_used"]), reverse=True)

        # Plain-English Insights Generator
        insights: List[str] = []

        # 1. Trigger insight
        if trigger_counts:
            top_triggers = list(trigger_counts.items())[:2]
            top_names = " and ".join([f"'{k}' ({v}x)" for k, v in top_triggers])
            insights.append(
                f"Primary vulnerability triggers for {child.first_name} are {top_names}, representing the leading catalyst for sensory overload."
            )

        # 2. Temporal & Location insight
        peak_time = max(time_dist.items(), key=lambda x: x[1])
        peak_location = max(location_counts.items(), key=lambda x: x[1]) if location_counts else ("home", 0)
        if peak_time[1] > 0:
            insights.append(
                f"Incidents cluster most frequently in the {peak_time[0].capitalize()} ({peak_time[1]} incidents), predominantly occurring at {peak_location[0].capitalize()}."
            )

        # 3. Strategy Efficacy insight
        if ranking:
            best_strat = ranking[0]
            insights.append(
                f"'{best_strat['strategy_name']}' demonstrated the highest de-escalation efficacy with a {best_strat['success_rate']}% success rate across {best_strat['times_used']} deployment(s)."
            )

        # 4. Duration and Intensity insight
        intensity_desc = "Mild" if avg_intensity <= 2.0 else ("Moderate" if avg_intensity <= 3.5 else "Severe")
        insights.append(
            f"Average incident duration is {avg_duration_min} minutes with an average intensity rating of {avg_intensity}/5 ({intensity_desc})."
        )

        return MeltdownAnalyticsResponse(
            child_id=child.id,
            child_name=child.first_name,
            total_incidents=total_incidents,
            avg_duration_minutes=avg_duration_min,
            avg_intensity=avg_intensity,
            trigger_frequencies=trigger_counts,
            location_frequencies=location_counts,
            time_of_day_distribution=time_dist,
            day_of_week_distribution=day_dist,
            strategy_efficacy_ranking=ranking,
            plain_english_insights=insights
        )
