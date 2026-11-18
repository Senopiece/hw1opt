#set document(title: "Robust cross-entropy under label noise", author: "Vitaly Makhonin", keywords: ("RobustCE", "label smoothing", "label noise", "cross-entropy"))
#set page(paper: "a4", margin: (top: 20mm, bottom: 20mm, left: 23mm, right: 23mm),
  header: align(right, text(size: 8pt, fill: luma(45%))[Robust cross-entropy under label noise]),
  footer: context align(center, text(size: 9pt, counter(page).display("1"))))
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: .58em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set figure(gap: .5em)
#show figure.caption: set text(size: 9pt)
#show heading.where(level: 1): set text(size: 15pt)
#show heading.where(level: 2): set text(size: 11.5pt)
#let fig(name, caption) = figure(image("assets/" + name + ".svg", width: 100%), caption: caption, placement: none)
#let note(body) = block(width: 100%, fill: luma(96%), inset: 9pt, radius: 2pt, body)
#let boxed(body) = box(stroke: .5pt + luma(50%), inset: 6pt, body)
#let data = json("assets/report_data.json")
#let result(dataset, method, eta) = data.main_results.find(r => r.dataset == dataset and r.method == method and r.eta == eta)
#let dec(x, digits: 4) = str(calc.round(x, digits: digits))

#align(center)[
  #v(4mm)
  #text(size: 23pt, weight: "bold")[Robust cross-entropy\ under label noise]
  #v(3mm)
  #text(size: 13pt)[Binary target uncertainty, smoothing,\ and a matched noise likelihood]
  #v(4mm)
  Vitaly Makhonin\
  Homework mini-project · 18 November 2026
]
#v(6mm)
#align(center, strong[Abstract])
We study binary cross-entropy maximized over an ellipsoidal target uncertainty set intersected with the probability simplex. In the binary case, this set becomes a clipped interval, giving a stable logits loss equal to ordinary cross-entropy plus a one-sided margin penalty. We verify the formula by direct numerical maximization, then compare CE, three robust radii, and label smoothing on Gaussian classes and noisy semicircles using CPU PyTorch.

Accuracy changes little. Robust training can improve a common-radius robust test objective, particularly with clean training labels, but ordinary CE has the lowest mean *clean test CE* in every main comparison. Gaussian parameter sweeps show the same pattern, with only tiny improvements near zero regularization on clean training data. Matched smoothing explains much of the robust loss's confidence reduction. A forward noise-aware likelihood, derived for the actual symmetric label-flip process, yields substantial clean-CE improvements when its assumed noise rate is appropriate.

#note[
  *Central finding.* Target-set robustness and robustness to randomly flipped hard labels are different problems. In these experiments, reducing confidence preserves much of the decision boundary but worsens nominal probability prediction. Explicitly modeling the corruption channel is more effective.
]

= Scope and reading order
The report follows the assignment's mathematical analysis, implementation, simulations, and critical analysis. Its narrative also follows the accompanying presentation: general geometry and the binary reduction (slides 3–28), numerical verification (slide 30), fair experiments (slides 31–36), the smoothing connection (slides 37–40), and the noise-aware follow-up (slide 42).

The main analysis emphasizes clean test CE after briefly reporting accuracy and common-radius robust CE. This is a choice of the prediction objective, rather than a claim that the robust score is an invalid metric. The final sections explain both the observed failures and a more closely matched future experiment. All empirical claims refer to the saved outputs of notebooks 01–06; no additional training was performed to prepare this report.

#pagebreak()
= Target uncertainty and the role of the simplex
== CE plus a constrained support term — assignment 1(a–b)
For predicted probabilities $p in Delta_K$, nominal target $mu in Delta_K$, positive definite $Sigma$, and radius $rho >= 0$, define
$
  cal(U)_Delta(mu,Sigma,rho) = { c in Delta_K : (c-mu)^T Sigma^(-1)(c-mu) <= rho^2 },
  quad L_"Robust"(p) = max_(c in cal(U)_Delta) (-c^T log p).
$
Logarithms act componentwise and are natural throughout. With $u=c-mu$, the presentation's CE-plus-penalty decomposition is
$
  L_"Robust"(p) = -mu^T log p + max_u (-u^T log p),
  quad u^T Sigma^(-1)u <= rho^2, quad bold(1)^T u=0, quad u>=-mu.
$
Both simplex constraints matter. Unit sum makes $c$ a normalized target and nonnegativity makes every class weight a valid probability. Dropping them gives the easier, but generally different, ellipsoidal support formula
$
  L_"ellipsoid"(p) = -mu^T log p + rho sqrt((log p)^T Sigma(log p)).
$
To derive this expression, substitute $u=Sigma^(1/2)w$ with $norm(w)<=rho$ and apply Cauchy–Schwarz to $-w^T Sigma^(1/2)log p$. Equality is attained at $u^*=-rho Sigma log p/sqrt((log p)^T Sigma(log p))$ when the denominator is nonzero. Its optimizer need not lie in the simplex. For example, if $p=(1/2,1/2)$, every valid target has CE $log 2$. The unconstrained ellipsoid with $Sigma=tau^2 I_2$ instead gives $log 2+rho tau sqrt(2) log 2$, strictly larger for positive radius. It cannot replace the simplex-constrained objective.

Since $mu$ is feasible, $L_"Robust">=L_"CE"$. At $rho=0$, the uncertainty set is the singleton ${mu}$ and the two losses coincide. Increasing $rho$ enlarges the set, so the robust loss is nondecreasing at fixed $p$. The same statements apply to the directly configured binary radius $delta$.

#fig("simplex_interval", [Binary geometry. The blue disk is the isotropic ellipsoid; the line segment is the probability simplex. Their intersection, shown in purple, is the feasible target interval. The hard-label case on the right is clipped at the simplex endpoint. The orange point is the nominal target.])

This is uncertainty *in the target distribution*, conditional on an input. It does not perturb the features, and it does not yet specify how an observed hard label was corrupted.

#pagebreak()
= Binary reduction, hand calculations, and convexity
== From an ellipsoid to an interval — assignment 1(c–d)
Write $p=(1-p_1,p_1)$, $p_1=op("sigmoid")(z)$, $mu=(1-m,m)$, and $c=(1-q,q)$. For $v=(-1,1)^T$, we have $c-mu=(q-m)v$. Therefore
$
  delta = rho / sqrt(v^T Sigma^(-1)v), quad
  q in [q_-,q_+]=[max(0,m-delta),min(1,m+delta)].
$
In particular, $Sigma=tau^2 I_2$ gives $delta=rho tau/sqrt(2)$. Only this product of scale and radius enters the binary interval. We set $delta$ directly in all experiments.

For a soft target $q$, stable binary CE is affine in that target:
$ L_"CE"(z,q)=op("softplus")(z)-q z. $
Thus $q^*=q_-$ for $z>0$, $q^*=q_+$ for $z<0$, and every feasible $q$ is optimal for $z=0$. Equivalently,
$
  L_"Robust"(z,m)=L_"CE"(z,m)
  +min(m,delta)op("ReLU")(z)+min(1-m,delta)op("ReLU")(-z).
$
For an observed hard label $y in {0,1}$, let $s=(2y-1)z$ be the signed margin and $hat(delta)=min(1,delta)$. The implemented loss is
$
  boxed(L_"Robust"(z,y)=op("softplus")(z)-y z+hat(delta)op("ReLU")(s)).
$
For $y=0$ and $delta=0.2$, the assignment's hand examples are:

#align(center, table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr), inset: 5pt, stroke: .4pt + luma(75%),
  table.header([$p=(p_0,p_1)$], [$z$], [$q^*$], [CE], [RobustCE]),
  [$(0.9,0.1)$], [−2.1972], [0.2], [0.1054], [0.5448],
  [$(0.6,0.4)$], [−0.4055], [0.2], [0.5108], [0.5919],
  [$(0.4,0.6)$], [+0.4055], [0], [0.9163], [0.9163],
))

The worst-case target shifts away from the predicted class when the observed label is already predicted correctly. For an incorrect prediction, the hard nominal target is itself worst-case; the robust penalty is then zero.

== Convexity — assignment 1(e)
The loss is convex in $p$ on the positive simplex: each feasible target produces a nonnegative weighted sum of convex functions $-log p_k$, and a pointwise maximum preserves convexity. It is also convex in binary logits, as CE plus a nonnegative ReLU penalty. Consequently it is convex in logistic-regression parameters, whose logits are affine. A tanh network's logits are nonlinear in its parameters, so its training objective is generally nonconvex.

#pagebreak()
= Stable implementation and direct numerical verification
== CPU PyTorch — assignment 2
Training uses logits directly, avoiding explicit logarithms of probabilities close to zero. For each example the code computes `softplus(z) - y*z`, then adds `min(1, delta) * relu((2*y-1)*z)`. The mean is differentiated with PyTorch autograd. At the ReLU kink, PyTorch's derivative of zero selects a valid subgradient; the inner target maximizer is nonunique there.

Notebook 01 independently evaluates `binary_cross_entropy_with_logits(z, q)` on *10,001 equally spaced feasible targets, including both endpoints*. It compares the maximum value with the analytic loss at
$
  y in {0,1}, quad z in {-8,-2,0,2,8}, quad
  delta in {0,0.05,0.2,0.5,1,1.5}.
$
All 60 float64 cases pass with absolute tolerance $10^(-10)$ and zero relative tolerance. The saved notebook reports maximum error $6.661 times 10^(-16)$. The report builder reruns this inexpensive verification and also checks the signs and CE baselines of the plotted comparisons. Since CE is affine in $q$, including the endpoints makes this grid check exact up to floating-point error; a dense grid alone would not generally guarantee an exact maximum for a nonlinear objective.

#fig("numerical_maximization", [Direct maximization for $delta=0.2$. The blue segment is the feasible interval, the orange dot is the numerical maximum, and the green dashed line is the analytic robust loss. At $z=0$, all feasible targets tie. This reproduces the verification shown on presentation slide 30.])

== Checks beyond the plotted inputs
The same parameter grid includes the CE limit $delta=0$, saturated uncertainty at $delta>=1$, positive and negative margins, and the zero-logit tie. The formula is nonnegative relative to nominal CE and nondecreasing in radius by construction. Negative radii are rejected.

Label smoothing uses $y_epsilon=(1-epsilon)y+epsilon/2$. The main comparison sets $epsilon=0.2$, matching RobustCE with $delta=0.1$ on positive margins. Section 11 derives exactly where the two objectives differ. Noise-aware likelihood and its numerical checks are described separately in Section 12.

#pagebreak()
= Simulation and fair comparison protocol
== Data, models, and shared randomness — assignment 3
*Gaussian classes.* Draw balanced labels and $x|y ~ cal(N)((2y-1)(1,1)^T,I_2)$. Fit affine logistic regression with three parameters. The population log odds are $2(x_1+x_2)$ in raw coordinates, so the clean conditional model is correctly specified by this architecture.

*Noisy semicircles.* Draw $t ~ U[0,pi]$, use $(cos t,sin t)$ for class 0 and $(1-cos t,0.5-sin t)$ for class 1, then add Gaussian noise with covariance $0.12 I_2$ (standard deviation $sqrt(0.12)$). Fit a 2–8–1 tanh network with 33 parameters.

Each seed produces independent balanced train/validation/test splits of 1,000/400/2,000 examples. Features are standardized using training statistics. Training labels are flipped independently with probability $eta$ *once per seed*; methods share the exact observed labels. Validation and test labels remain clean. Methods also share data splits and initial weights. Noise levels use shared uniforms, so increasing $eta$ nests the corruption masks. Seeds vary both the sampled data and model initialization. These small models and full-batch budgets make paired repeated experiments inexpensive, while independent validation and a larger test split support probability-score comparisons.

== Main experiment — assignment 4
Compare CE, RobustCE with $delta in {0.05,0.1,0.2}$, and smoothing with $epsilon=0.2$. Use float32 CPU PyTorch 2.11.0, one thread, full-batch Adam with default betas/epsilon, no weight decay or schedule, and 400 epochs for Gaussian or 800 for moons. Select the *last epoch*, with no early stopping or checkpoint selection on test data.

Tune only the learning rate over ${0.001,0.01,0.05}$, separately for each dataset and objective, using mean final *clean validation CE* over seeds 0, 1, 2. Training noise during this selection is fixed at $eta=0.1$. Freeze the choices and retrain from scratch at both $eta=0$ and $eta=0.1$ using new seeds 101–105. This intentionally differs from the assignment's initial clean-training tuning prescription; it follows the fixed protocol used in the presentation.

#fig("lr_selection", [Mean clean validation CE ± sample SD across three tuning seeds; stars mark the chosen rate. All five Gaussian methods choose 0.05, the largest tested value. All moons methods choose 0.01. An endpoint choice limits the search claim: larger Gaussian rates were not tested.])

The main experiment contains 90 tuning and 100 evaluation fits, with zero reported failures. Saved wall times were 33.9 s and 37.7 s, respectively. Mean training time per fit was 0.167 s for Gaussian CE versus 0.192–0.202 s for robust variants; for moons, 0.527 s versus 0.572–0.582 s. Smoothing took 0.172 s and 0.528 s. These are environment-specific timings, not a general performance benchmark.

#pagebreak()
= Which test metric answers the question?
== Brief accuracy and robust-risk check
All methods are evaluated on the same clean test labels using accuracy, ordinary CE, and robust CE with a *common evaluation radius* $delta_"eval"=0.1$. The latter is centered at each clean test label, regardless of training objective or training radius.

#fig("auxiliary_metrics", [Auxiliary metrics, mean ± sample SD over five evaluation seeds. Accuracy is similar across losses. With clean training, robust and smoothed models improve the common-radius robust score; with corrupted training this advantage largely disappears. A lower robust score alone does not establish better nominal prediction or calibration.])

Across the main comparisons, the entire between-method accuracy spread is at most 0.27 percentage points within a dataset/noise condition, small relative to the observed seed variability. Accuracy mainly assesses the side of the $p=0.5$ boundary and discards confidence. Robust CE assesses a different, deliberately conservative target-uncertainty risk. It is meaningful when that risk is the task, but is closely aligned with the robust training objective and is not a neutral primary criterion for nominal probabilities. Calling it a statistically biased estimator merely because of this alignment would be inaccurate.

== Clean CE and a precise connection to mutual information
For the clean population distribution $P$ and model probabilities $q_theta$, define $C(theta)=bb(E)[-log q_theta(Y|X)]$. Then
$
  C(theta)=H(Y|X)+bb(E)_X D_"KL"(P(Y|X) parallel q_theta(Y|X)),
$
$
  H(Y)-C(theta)=I(X;Y)-bb(E)_X D_"KL"(P(Y|X) parallel q_theta(Y|X)) <= I(X;Y).
$
Thus lower clean CE tightens a variational lower bound on MI, in the sense of Barber and Agakov [1]. For our balanced population $H(Y)=log 2$ nats. Training changes this bound through the model's conditional approximation; it does *not* change the true MI of the fixed data-generating process. An empirical test estimate also has sampling error and need not itself be a lower bound in every finite sample. We therefore use clean test CE as the primary score for the remaining empirical analysis.

#pagebreak()
= Main result: ordinary CE predicts clean probabilities best
#fig("clean_ce", [Primary comparison: ordinary clean test CE, lower is better. Points are means and bars are sample SD across the same five seeds. Every nonzero robust radius and the matched smoothing coefficient have higher mean CE than ordinary CE under both clean and corrupted training.])

On Gaussian data, CE changes from 0.2006 with clean training to 0.2795 at $eta=0.1$. RobustCE(0.1) instead gives 0.2767 and 0.3651; smoothing(0.2) gives 0.2743 and 0.3527. On moons, CE gives 0.2807 and 0.3199, compared with 0.3215 and 0.3852 for RobustCE(0.1), and 0.3206 and 0.3746 for smoothing.

Increasing the robust radius to 0.2 pushes clean test CE still higher: approximately 0.3725/0.4545 on Gaussian and 0.3976/0.4654 on moons for clean/corrupted training. The smaller radius 0.05 is less harmful, but still does not beat CE in any main condition. These are empirical results for the specified models and training budget, rather than a universal ordering of learning algorithms.

The MI interpretation provides the same ordering without adding another empirical metric: for clean-trained Gaussian models, $log 2-C$ is approximately 0.4926 nats for CE and 0.4165 for RobustCE(0.1). These numbers estimate model-dependent lower-bound values; they are not estimates of two different true mutual informations.

== Why this result is plausible
For Gaussian classes, clean logistic regression already contains the true conditional probabilities. Penalizing positive margins therefore introduces a systematic incentive to become less confident even where confidence is justified. Under symmetric label noise, ordinary CE already fits a compressed observed-label posterior; adding another confidence-reducing objective can compound that compression.

For moons, the nonlinear model could in principle benefit from regularization, so the Gaussian argument is not a proof of failure. Nevertheless, the same ordering is observed. We examine learning curves and probability maps next to separate changes in confidence from changes in classification boundaries.

#fig("radius_validation", [Effect of robust training radius on final clean validation CE after LR selection, training at $eta=0.1$. The zero-radius point is CE. Bars show sample SD over the three tuning seeds. The worsening validation score agrees with the independent main test comparison; the radius was not chosen by inspecting test losses.])

#pagebreak()
= Confidence reduction and training dynamics
#fig("learning_curves", [Clean validation CE throughout training for representative seed 101. This is one common metric for all training losses, rather than a comparison of different training-objective values. Curves are sampled every 20 epochs; evaluation always uses the final epoch.])

#fig("confidence", [Mean clean-test confidence $bb(E)[max(p_theta,1-p_theta)]$, with sample SD over five seeds. Larger radii consistently lower confidence. Confidence is not a calibration measure: lower values can help or hurt depending on the true conditional probabilities.])

The main curves level off at different clean-CE values rather than merely showing one method learning more slowly. Within the tested budget, the robust variants usually settle at more conservative probabilities. For Gaussian clean training, confidence falls from about 92.1% with CE to 80.4% at $delta=0.1$ and 71.6% at $delta=0.2$, while accuracy remains near 91.7%.

This behavior is predicted by the one-example objective. For $0<delta<1/2$, minimizing $op("softplus")(-s)+delta op("ReLU")(s)$ gives
$ s^*=log((1-delta)/delta), quad op("sigmoid")(s^*)=1-delta. $
For $delta>=1/2$, the minimum is at $s=0$. These are per-example optima, not hard probability caps in a shared-parameter model. They explain the pressure toward smaller margins, not a theorem that every prediction has exactly confidence $1-delta$.

#pagebreak()
= Similar boundaries, less certainty
#fig("boundaries_clean", [Clean training ($eta=0$), representative seed 101. Black curves are the $p=0.5$ boundaries; colors encode the same predicted class-1 probability in every panel. Points are 200 clean test examples. Coordinates use training-feature standardization.])

#fig("boundaries_noisy", [Corrupted training ($eta=0.1$), the same representative seed and plotting convention. CE, RobustCE(0.1), and matched smoothing retain broadly similar boundaries, while the latter two produce less saturated probabilities.])

These maps support a qualified version of the presentation's interpretation: the main effect is to flatten predicted probabilities while retaining much of the fitted boundary. They do not prove that the boundaries are identical, nor that confidence reduction is always harmful. Here the clean log score shows that, averaged over the held-out distribution, the reduction is excessive. The result also explains how accuracy can remain stable while clean CE worsens substantially.

#pagebreak()
= Gaussian sweeps: is the conclusion just a bad coefficient?
The follow-on experiments use $eta in {0,0.025,0.05,0.1,0.2,0.3,0.4}$ and the 12 strengths shown below. For each noise/strength cell, independently choose the learning rate from the same three values using clean validation CE and tuning seeds 0–2. Evaluate with new seeds 201–205, sharing data, initial weights, and flips across methods. These exploratory sweeps retune LR at each noise level; the main experiment in Section 5 keeps it frozen across its two noise levels.

#fig("robust_grid", [Robust sweep: $C_"CE"-C_"Robust"$ on clean test labels. Green favors RobustCE, red favors CE, and white is zero or a small absolute change. Each cell is a mean paired difference over five seeds. Columns are categorical, not equally spaced numeric intervals.])

#fig("smoothing_grid", [Smoothing sweep: $C_"CE"-C_"Smooth"$. The color scale is identical to the robust panel. The strengths satisfy $epsilon=2delta$ column by column. Zero-strength controls reproduce CE.])

Each sweep has 756 tuning and 420 evaluation records, with no reported failures; compatible CE controls are reusable. Neither method wins at any tested *positive* corruption rate. On clean training there are three tiny positive cells per method: $delta=0.0001,0.001,0.0025$ and $epsilon=0.0002,0.002,0.005$. The largest gains are only 0.000128 and 0.000134 nats, respectively, under 0.07% relative CE reduction. Each of these cells wins on 4/5 seeds, and its paired sample SD exceeds its mean gain. They qualify an absolute “never helps” claim, but provide no convincing practical improvement. The many exploratory cell comparisons are not significance tests.

#pagebreak()
= Smoothing and robust CE: the same positive-margin penalty
For $y_epsilon=(1-epsilon)y+epsilon/2$ and $s=(2y-1)z$,
$
  L_"Smooth"=L_"CE"+epsilon/2 s, quad
  L_"Robust"=L_"CE"+hat(delta)op("ReLU")(s).
$
When $epsilon=2hat(delta)<=1$, the presentation's identity is exact:
$
  boxed(L_"Robust"=max(L_"CE",L_"Smooth")), quad
  L_"Robust"-L_"Smooth"=hat(delta)op("ReLU")(-s).
$
On positive margins the objectives coincide. On negative margins smoothing lowers the nominal CE penalty, whereas RobustCE keeps the hard-label CE unchanged. At $delta=0.1$, matching requires $epsilon=0.2$ and gives the smoothed observed-label target probability 0.9.

#fig("margin_losses", [Analytic comparison at matched coefficients. The robust objective takes the larger of nominal CE and smoothed CE. Its distinctive worst-case construction acts on the negative-margin difference; it is not a different penalty on already correct positive-margin examples.])

#fig("matched_grid", [Direct matched comparison: $C_"Smooth"-C_"Robust"$ on clean test labels. Positive/green favors RobustCE; negative/red favors smoothing. Each method retains its independently validation-selected LR. The comparison reuses the verified sweep outputs and requires no new training.])

Smoothing has lower mean CE in 69 cells, RobustCE in 6, with 9 ties at tolerance $10^(-6)$. All six robust wins occur at the degenerate edge $delta=0.5$, $epsilon=1$. Both losses then prefer a zero per-example margin, but finite-budget optimization of the nonsmooth robust objective leaves residual confidence. These wins over an almost uniform smoothing model are not evidence that RobustCE beats ordinary CE. At moderate strengths, the matched smoothing model is generally slightly better, and neither delivers the desired improvement over CE.

#pagebreak()
= A direct alternative: likelihood of the corruption channel
The failure under label flips suggests modeling the observation mechanism rather than introducing target-set uncertainty. Let $Y$ be the clean label and $tilde(Y)$ the observed training label. Under independent symmetric flips with an assumed rate $alpha=eta_"expected"<1/2$,
$
  T_alpha=mat(1-alpha,alpha;alpha,1-alpha), quad
  P(tilde(Y)=1|X)=q_theta=alpha+(1-2alpha)p_theta,
$
where $p_theta=op("sigmoid")(z)$ models the *clean* class-1 probability. The forward noise-aware CE is
$
  L_"NA"(z,tilde(y);alpha)=-tilde(y)log q_theta-(1-tilde(y))log(1-q_theta).
$
This is forward loss correction in the sense of Patrini et al. [2], specialized to a binary symmetric channel. Training scores the noisy observation under $q_theta$; clean evaluation scores $p_theta$ against $Y$. Evaluating the noisy posterior $q_theta$ on clean test labels would answer a different question.

#fig("noise_channel", [The model predicts a clean probability, then the known or assumed channel maps it to an observed-label probability used by the training likelihood. The channel is absent from clean test evaluation.])

== Stable form and the CE limit
With $tilde(s)=(2tilde(y)-1)z$, the observed-label probability is $(1-alpha)op("sigmoid")(tilde(s))+alpha op("sigmoid")(-tilde(s))$. Compute the loss in log space:
$
  L_"NA"=-op("logaddexp")(
    log(1-alpha)+op("logsigmoid")(tilde(s)),
    log alpha+op("logsigmoid")(-tilde(s))
  ).
$
At $alpha=0$, the implementation uses exactly the ordinary CE branch. Values outside $[0,1/2)$ are rejected: at $1/2$ the channel erases all label information. Notebook 06 checks agreement of values and gradients with direct probability-space BCE at moderate logits, the zero-noise limit, and finiteness for logits as large as $plus.minus 1000$.

== Why this should help under the simulated noise
If the true clean posterior is $r(x)$, ordinary CE fitted to noisy labels targets $eta+(1-2eta)r(x)$, which is compressed toward $1/2$. When $alpha=eta<1/2$, the noise-aware likelihood is minimized at $p_theta=r(x)$, provided the model can represent it. This is a population statement, not a finite-sample guarantee. It is particularly appropriate for the Gaussian logistic model, whose clean posterior is correctly specified. The actual noise rate is known from the simulation; estimating it from imperfect real data is a separate problem.

#pagebreak()
= Noise-aware results: correcting the matched process helps
Use the same Gaussian splits, optimizer, budgets, tuning seeds 0–2 and evaluation seeds 201–205 as in the parameter sweeps. Cross the seven actual noise rates with 12 assumed rates $alpha in {0,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.45}$. LR is chosen by clean validation CE for each cell. This again gives 756 tuning and 420 evaluation records, with zero reported failures.

#fig("noise_aware_grid", [Clean test $C_"CE"-C_"NA"$: green favors noise-aware CE, red favors ordinary CE. Outlined cells use the true generating rate, $alpha=eta$; they are specified by the simulator, not selected by test performance. The zero assumed-noise column is the CE control.])

#fig("noise_aware_diagonal", [Correctly specified channel, $alpha=eta$. Left: mean clean test CE. Right: paired CE reduction ± sample SD over five seeds. All nonzero-noise diagonal cells improve on CE in all five seeds. The observed improvement becomes large as the label-flip rate rises.])

At $eta=alpha=0.1$, clean CE falls from 0.26635 to 0.19279, a reduction of 0.07356 nats (about 27.6%). At 0.2 it falls from 0.36119 to 0.19326, and at 0.4 from 0.56747 to 0.25301. These numbers use the sweep seeds and must not be directly subtracted from the main experiment's different-seed results.

Across the full grid, noise-aware CE wins in 63 cells, CE in 14, with 7 exact control ties. Misspecification matters: assuming positive noise when labels are clean makes CE worse. The benefit therefore supports a matched observation model, not a claim of universal superiority. Learning an unknown noise rate, trying other corruption mechanisms, or extending this follow-up to moons was not part of the completed experiments.

#pagebreak()
= Critical analysis — assignment 5(a–e)
== (a) Where does RobustCE help, and where does it fail?
It helps the *common-radius robust test objective* with clean training, but not the primary nominal clean-CE objective in any main comparison. Accuracy is nearly unchanged. Gaussian sweeps reveal only negligible near-zero-strength improvements on clean training, with no wins at positive tested label noise. A larger robust training loss than CE is not itself evidence of a worse fitted model: the objectives differ by a nonnegative penalty. The held-out common metrics establish the relevant comparison.

== (b) Radius, confidence, and conservatism
Increasing $delta$ lowers confidence and raises clean CE in the main experiments. The three radii 0.05, 0.1, and 0.2 show a consistent progression; the per-example optimum $s^*=log((1-delta)/delta)$ explains it. Robustness to uncertainty trades against fitting confident clean probabilities. Here this trade is excessive even though the boundary remains useful. Confidence plots support the mechanism, but a calibration claim would require a separate analysis, such as reliability curves or a proper calibration decomposition.

== (c) Does a clean-training benefit persist under corrupted labels?
For nominal clean CE there is no substantial clean-training benefit to preserve. For robust test CE, the improvement seen under clean training does not persist clearly at $eta=0.1$. This answers the conditional premise of part (c), rather than treating a nonexistent nominal gain as given. The worst-case target construction also explains why label flips are not automatically repaired: on a negative observed-label margin it retains ordinary CE's full pressure to fit that label. It does not identify or discard a mislabeled example. On a positive margin it discourages certainty regardless of whether that observed label was clean.

== (d) What is specific to worst-case training beyond smoothing?
At $epsilon=2delta$, RobustCE is the upper envelope of CE and smoothing. Its unique feature is preserving the hard-label penalty on negative margins. The matched experiments do not show this distinction yielding better clean probabilities than smoothing at moderate strengths. Most of the observed confidence effect can be explained by the shared positive-margin penalty. Evidence of a specific worst-case advantage would require a reproducible gain over matched smoothing on a predeclared target-uncertainty test, under comparable tuning and training budgets, with nominal-CE tradeoffs reported. A win over CE alone would not isolate that advantage. These experiments do not establish equivalence of the methods in general.

== (e) Limitations and a focused follow-up
The models are small and the noise is synthetic, independent, and symmetric. There are only five evaluation seeds; many exploratory grid cells, a finite LR search, and endpoint selections limit generalization and inference. The clean validation set is an important advantage, especially for method and LR selection. No robust radius was calibrated to a statistical confidence region, and lower robust test CE was not validated as a surrogate for nominal CE or calibration.

*Proposed next experiment: match the corruption to target uncertainty.* Generate known nominal soft targets $m(x)$, give each an interval $[max(0,m-delta),min(1,m+delta)]$, and evaluate an explicit shift to soft targets inside that interval, including the worst-case endpoints. Train CE, smoothing, and the corresponding soft-target robust loss under the same budget, then report both nominal CE and the predeclared worst-case soft-target CE. This would test the risk RobustCE actually optimizes. It is a future experiment, not an explanation retroactively verified by the present hard-label flips.

#pagebreak()
= Reproducibility, assignment coverage, and references
== Runnable artifacts
Dependencies are managed with `uv`. From the repository root:
```powershell
uv sync --locked
uv run --locked python docs/build_report.py
```
The report builder reads the exact numeric Plotly outputs already saved in the notebooks, redraws the scientific figures as vector SVG, reruns the direct loss check, validates the comparison signs, and compiles `docs/report.typ` to `docs/report.pdf`. It does not retrain models or require optional experiment caches. `docs/assets/report_data.json` records the extracted statistics and SHA-256 hashes of the six source notebooks. To regenerate experimental results, execute the notebooks in order using the commands in the repository README; the sweeps have resumable caches and provenance checks.

#table(
  columns: (auto, 1fr, 1fr), inset: 5pt, stroke: .4pt + luma(78%),
  table.header([*Notebook*], [*Purpose*], [*Report coverage*]),
  [01], [Analytic loss vs numerical target maximization], [Sections 2–4],
  [02], [Two datasets, five losses, clean/noisy training], [Sections 5–9],
  [03 / 04], [Gaussian radius and smoothing sweeps], [Section 10],
  [05], [Matched smoothing–robust comparison], [Section 11],
  [06], [Noise-aware derivation and assumed-noise sweep], [Sections 12–13],
)

== Coverage and deliberate departures
Assignment 1(a–e) is covered in Sections 2–3, including the simplex constraint, limiting cases, binary maximizer, hand calculations, and convexity. Assignment 2 is covered in Section 4 and notebook 01. Assignments 3–4 are covered by the shared simulation protocol, LR sensitivity, learning curves, three held-out metrics, confidence, boundaries, repeated seeds, failures, and timings. Section 14 answers critical analysis 5(a–e) explicitly. The report and six runnable notebooks supply the requested written and computational artifacts.

The principal departures are explicit: main LR tuning uses training noise 0.1 rather than initial clean training; fixed main loss strengths are supplemented by independent Gaussian sweeps; and most interpretation after the initial metric check uses nominal clean test CE. The extra noise-aware experiment goes beyond the assignment to test an alternative tailored to the actual corruption mechanism. This ordering matches the supplied presentation while retaining its missing protocol details and qualifications.

== Contribution and evidence boundary
The accompanying presentation lists Vitaly Makhonin as its author. Project work covered the formulation and binary derivation, CPU PyTorch implementation and verification, experimental design, analysis, and presentation. All reported model comparisons come from the saved notebook experiment snapshot. The geometry and channel figures are explanatory constructions, and the proposed soft-target corruption study has not been run.

== References
#set text(size: 9.5pt)
[1] D. Barber and F. V. Agakov. “The IM Algorithm: A Variational Approach to Information Maximization.” _Advances in Neural Information Processing Systems 16_, 2003. #link("https://papers.nips.cc/paper_files/paper/2003/file/a6ea8471c120fe8cc35a2954c9b9c595-Paper.pdf")[Original paper.]

[2] G. Patrini, A. Rozza, A. K. Menon, R. Nock, and L. Qu. “Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach.” _CVPR_, pp. 1944–1952, 2017. #link("https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html")[Paper and proceedings.]

[3] _Homework Mini-Project: Robust Cross-Entropy over Ellipsoidal Targets._ Supplied assignment, `HW_robust_ce.pdf`.

[4] V. Makhonin. _Homework Mini-Project._ Supplied 43-slide presentation, `docs/presentation.pdf`.
