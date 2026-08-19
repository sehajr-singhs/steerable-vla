"""Steerable VLA — a miniature, fully runnable instantiation of the proposal.

Architecture implemented here (toy scale, faithful in structure):

  env        CableEnv           2D mass-spring cable, pinned ends, crossings,
                                gripper point; procedural crossing/stiffness
                                families (the "chaotic task" analog)
  policy     FlowExpert         conditional flow matching over action chunks
                                with subgoal conditioning (state keyframes)
  steering   SMC adapter        gated steering branch on the velocity field
                                with anchoring at u = 0 (neutral)
  safety     CBF-QP filter      forward-invariant projection of the commanded
                                action onto the admissible set
  flywheel   loop               deploy -> score -> curate -> relabel -> retrain
  eval       harness            pre-registered held-out families, 3 seeds,
                                metrics: success, crossings reduced, jerk,
                                violations, interventions

Every number in the papers/site regenerates from results/*.json committed here.
"""

__version__ = "0.1.0"
