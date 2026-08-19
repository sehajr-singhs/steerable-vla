# Statement of Purpose — Research Scientist Internship, Physical Intelligence

*Sehaj Singh · Independent Research · 2026*

---

The gap between a policy that works in a lab and a policy that works in a
warehouse is not model scale. It is not data volume. It is the ability to take a
long-horizon task, decompose it under uncertainty, and execute the pieces at
control frequency without breaking anything when the world moves — which it
always does.

I want to spend this internship working on that gap, inside the π-series stack.

What I bring is not a claim that I know the answer. It is evidence that I have
been asking the right questions, at a scale I could afford, with results that
survive scrutiny.

**The data side, first.** My most recent published work is a controlled study of
the robot data flywheel — the closed loop in which a deployed policy's rollouts
are collected, curated, and fed back into retraining. Every serious scaling bet in
this field — RT-2, Open X-Embodiment, DROID, and the deployment programs that
followed — assumes that loop compounds. I built the smallest faithful
implementation and varied only the curation strategy. The result: with no
feedback, a frozen policy sits at 0.12 held-out success and never moves.
Self-curation (keeping your own successes) plateaus — it densifies what you
already do. Relabeling deployment failures with an oracle compounds: 0.15 → 0.66,
four times the success, at a fraction of the interaction budget of RL from
scratch. Curated relabeling bought 1.4× the performance per label, and it was
robust to label noise in a way blind relabeling was not. Every number in that
paper regenerates from a committed JSON. I did not claim the result; I committed
it.

That finding is a thesis about *why* scaling works in this field, and it has a
direct consequence for π0-class models: the marginal value of an episode depends
on the curation rule, and the highest-leverage investment in the π stack is not
only more compute — it is a scoring layer that decides which deployment failures
deserve an expensive label, and an oracle channel that can produce it. I would
like to spend part of this internship testing that at your scale, where rollout
telemetry is orders of magnitude richer than anything I could generate.

**The policy side.** I have trained and deployed diffusion policies and VLA
architectures end-to-end: data collection and teleoperation pipelines, action
chunking with receding-horizon execution, sim-to-real with domain randomization,
and the less glamorous engineering — calibration, gripper failure modes, the
intervention rate as a number you actually track rather than a hope. On real
hardware I have learned the metric that papers underreport: not how often a policy
succeeds, but how often a human has to stop it. That number is the operating
cost of an embodied system, and it is the number I design against.

**Why Physical Intelligence, and why the π-series specifically.** Because π0 got
two architectural decisions right that most of the field has still not internalized.
First, *flow matching over diffusion* for action generation: few-step sampling that
keeps trajectories smooth at control frequency, in a formulation that is an ODE —
a velocity field you can analyze, steer, and certify, not a stochastic sampler you
can only sample from. Second, *the VLM as the hierarchy*: language and visual
subgoal tokens flowing through the same backbone that produces actions, so
semantic reasoning and motor control share one learned representation instead of
two models awkwardly glued together. π0.7 scaling that across embodiments and
object types is, to my reading, the strongest public evidence that this
parameterization is the right substrate for generalist manipulation.

The proposal I am attaching ([proposal.md](proposal.md)) is what I would do with
an internship: add the third decision. A **Steerable Multimodal Conditioning
layer** — a gated adapter that lets human corrections, world-model replans, and
constraint setpoints steer the flow field mid-execution, with a Grönwall-type
bound certifying that steering cannot jerk the robot — wrapped in a
**control-barrier-function safety envelope** that admits steerability only through
a formally verified filter. I chose the three benchmark tasks (cable untangling,
textile folding on shifting fabric, adaptive tool use in clutter) to be the ones
that defeat both hardcoded logic and monolithic imitation: non-rigid,
high-entropy, and long-horizon, exactly where subgoal factorization and
steerability stop being conveniences and become necessary.

What I am not asking for: a project to be handed to me, or credit for ideas
someone else already had. I am asking for access to the hardest version of a
problem I have already started solving, and for the chance to be wrong in public,
quickly, in a place where being wrong is cheap and the data loop is real.

I build things that compile, measure things I can defend, and write papers where
every number traces to a committed artifact. Give me a bench, a teleop rig, and
the π stack, and I will give you back a steerable, verifiable action expert and
the ablation study that says whether it was worth it.
