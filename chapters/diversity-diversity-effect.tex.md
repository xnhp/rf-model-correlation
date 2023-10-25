\documentclass[../main.tex]{subfiles}
\begin{document}

chapter introduction. We will now introduce a way to understand ensemble diversity. 

\section{Measures of ensemble diversity} \label{sec:measures-ensemble-diversity}

Review.  Maybe a table with all these measures? Would look nice. But probably not too relevant. 


% TODO write somewhere that bayes error can also be considered a variance (see Adlam22 eq (5)
\section{Ambiguity} 

% TODO cite 
%- krog_NeuralNetworkEnsembles
% - wood23

As we have seen in~\ref{sec:measures-ensemble-diversity}, formally expressing the notion of ensemble diversity is not straightforward. 

% TODO clarify that were omitting E_(X,Y) here
We take inspiration from~\ref{sec:bias-variance-effects} and derive a measure of ensemble diversity by considering the \textit{effect} of diversity on the ensemble error.
% TODO in other places, randomness of models is within RV D, here we use Theta
If we consider the members to be constructed according to a parameter $\Theta$, a reasonable measure of the member performance is its loss in expectation over the parameter distribution: $\mathbb{E}_{\Theta}\left[ L(Y, q(X; \Theta))  \right] \approx \Mavg L(Y, q(X;\Theta_{i})$.
% TODO make clearer that we're actually assuming homogeneous
What can we hope to gain from using an ensemble $\bar{q}$, which combines the output of individual members $q_{1}, \dots, q_{M}$ over just using a single member model? 
The \textit{ensemble improvement} for finite ensembles is 
\marginnote{At this point, it is not yet clear whether the ensemble improvement is even nonnegative, i.e. that ensembling does not hurt performance. We will address this in \ref{sec:todo}.}
$$
\Mavg L(Y, q_{i}) - L(Y, \bar{q})
$$
Conveniently, this is also the \textit{effect of ensembling} on the error. Due to this, we will refer to this quantity as \textit{ambiguity-effect}.
% TODO some sentence on how this was defined as ambiguity-effect in krogh or sth

\marginfigfromsource{fig:ambiguity-spread}{symlinks/illustrations/ambiguity/rf-ambiguity}

% TODO wood23 motivate this from perspective that ambig-eff is special case of bias-variance decomp

% \begin{marginfigure} \label{fig:ambiguity-spread}
%     \includegraphics[width=\textwidth]{symlinks/illustrations/ambiguity/rf-ambiguity.png}
%     \input{symlinks/illustrations/ambiguity/rf-ambiguity.tex}
% \end{marginfigure}

\begin{theorem} \label{thm:ambig-effect-decomp} (Ambiguity-Effect decomposition \cite{todo}) For any loss function $L$, target label $Y$, ensemble members $q_{1}, \dots, q_{M}$ with combiner $\bar{q}$
$$
L(Y, \bar{q}) = \Mavg L(Y, q_{i}) -
\underbrace{\left(\Mavg L(Y, q_{i}) - L(Y, \bar{q})\right)}_{\text{Ambiguity-Effect / Ensemble Improvement}}
$$
% TODO use LE notation
\end{theorem}

%\begin{theorem} \label{thm:ambiguity-decomp} (Ambiguity decomposition \cite{todo:krogh}) For any of the loss functions of \ref{thm:the-trick}, the ambiguity-effect term reduces to the target-independent \textit{ambiguity}
%\begin{align*}
%L(Y, \bar{q}) &= \Mavg L(Y, q_{i}) - \left(\Mavg L(Y, q_{i}) - L(Y, \bar{q})\right) \\
% &= \Mavg L(Y, q_{i}) - \Mavg L(q_{i}, \bar{q})
%\end{align*}
%\end{theorem}
% TODO has been discussed in the literature earlier

Similar to variance, ambiguity and ambiguity-effect are measures of spread. 
Variance measures the spread of training error across models trained with different draws of the training dataset $D$ around a model that is a centroid with respect to the distribution of $D$. Similarly, ambiguity measures the spread of individual member model errors around a model that is centroid with respect to the distribution of $\Theta$, namely the combiner $\bar{q}$. In case of squared loss, this is indeed the statistical variance. For other losses, this is a different quantity.
% can maybe move the above into a side note.



%TODO need to argue more that this is indeed centroid? should come from generalised / bregman etc.?
\section{The Diversity-Effect decomposition} \label{sec:bias-variance-diversity-decomp}
% TODO double decomposition trick and bias-variance-diversity/ambiguity-effect decomp

Note that the first term in the ambiguity-effect decomposition \ref{thm:ambig-effect-decomp} is the average loss of the ensemble members. One can now decompose the loss of an individual ensemble member into bias- and variance-effect just like in \ref{thm:bias-variance-effect-decomp}.
% TODO elaborate a bit more?

\begin{marginfigure}
    \includegraphics[width=\textwidth]{symlinks/zero-one-plots/bvd-decomps/spambase-openml/bvd-individual/standard-rf-classifier}
    \label{fig:spambase-standard-rf-classifier-bvd}
    \caption{
        foo bar baz, avg bias, variance constant, diversity increases
    }
    % TODO legend?
    % TODO here, it would actually be good to increase x axis range to show that the entire thing stabilises
\end{marginfigure}

% TODO maybe write like ... = (noise) + (average bias) + ...; where (average bias) = ..., probably easier to read

\begin{theorem} (Bias-Variance-Diversity-Effect decomposition)
\begin{align*}
    \mathbb{E}_{D,Y}\left[ L(Y, \bar{q}) \right] &= 
    \underbrace{ \mathbb{E}_{y}\left[ l(y, y^\star) \right] }_{\text{noise}} \\
    &+
    % \mathbb{E}_{\Theta, Y}\left[ L(Y, q_{\Theta}^\star) - L(Y, y^\star)\right] % avg bias-effect
    \underbrace{
        \frac{1}{M} \sum_{i=1}^M \mathbb{E}_Y\left[L_{0 / 1}\left(Y, q_i^*\right)-L_{0 / 1}\left(Y, Y^*\right)\right]
    }_{\overline{\text{bias-effect}}} \\
    &+
    \underbrace{
        \frac{1}{M} \sum_{i=1}^M \mathbb{E}_D\left[\mathbb{E}_Y\left[L_{0 / 1}\left(Y, q_i\right)-L\left(Y, q_i^*\right)\right]\right]
    }_{\overline{\text{variance-effect}}} \\
    % \mathbb{E}_{\Theta, D, Y}\left[ L(Y, q_{\Theta}) - L(Y, q_{\Theta}^\star) \right]  % avg variance-effect
    &- % minus!!
    % \mathbb{E}_{D, Y, \Theta}\left[ L(Y, q_{\Theta} - L(Y, \bar{q})) \right] % avg div-effect
    \underbrace{
    \mathbb{E}_D\left[\mathbb{E}_Y\left[\frac{1}{M} \sum_{i=1}^M\left[L_{0 / 1}\left(Y, q_i\right)-L(Y, \bar{q})\right]\right]\right]
    }_{\text{diversity-effect}}
\end{align*}
% TODO use LE notation
\end{theorem}



\section{Diversity for Bregman Divergences} \label{sec:bregman-divergences}

% TODO super interesting: bregman information, is just sample variance for squared loss, mutual information for KL-divergence, also interpretable for Itakura-Saito -- so here it sort of also appears in the purity measures
% https://jmlr.csail.mit.edu/papers/volume6/banerjee05b/banerjee05b.pdf
% also note their application of Jense's inequality: the difference between the two sides is exactly equal to the bregman information
% -- could this be the ensemble improvement?!??!
% ah yes, okay, we can already sort of read that between the lines from what we already have

% TODO somewhere: table with common loss funcitons, their generators, combiners

% TODO review on breggies: applications e.g. in generative modelling, GANs? 
% chatbot says:
% Good question. One field where maximizing the generator function can be important is in generative modeling/unsupervised learning tasks.
% Specifically, in generative adversarial networks (GANs):
% - GANs aim to learn a generative model G that produces samples resembling the real data distribution Pdata.
% - G competes against a discriminative model D whose goal is to distinguish real from generated samples. 
% - D's training objective is to maximize the probability of assigning the correct label (real vs fake) to both real and generated samples.
% - In other words, D aims to maximize the divergence metric (usually JS divergence) between Pdata and Pg (the generated distribution).
% - Maximizing this divergence encourages D to become better at discrimination, which provides stronger feedback to help G improve.
% So in GAN training, the discriminator D plays the role of maximizing the generator function (divergence metric), in order to facilitate learning the best generative model G. Maximizing the divergence is important for the adversarial process that improves both models.
% Other areas like domain adaptation may also involve maximizing divergences between distributions to learn robust features/mappings across domains.

Note that bias-effect, variance-effect and ambiguity-effect are all of the form $\mathbb{E}_{}\left[ L(Y, \circ) - L(Y, \square) \right]$ and depend directly on the target label $Y$. 
With variance-effect, we have captured the \textit{effect} of variations between different training datasets) on the prediction error.
With ambiguity-effect, we have captured the effect of variations between the different member models on the prediction error.
We have already seen that for the squared error, the variance-effect coincides with familiar notion of statistical variance between the predictions.
This is convenient, since we can then estimate variance (and its effect) independently of the target label $Y$.
This means that variance can be compared across different datasets.
% TODO re-check that this is true
The same applies to ambiguity \sidenote{Bias will always be indirectly dependent on the target labels via the expected label.}.
Also, labelled data can be scarce in practise, while examples alone may be easier to obtain. 

We will now define a class of losses for which the effects reduce to the quantities themselves. This class covers many widely used loss functions and thus allows us to formulate a unified bias-variance-decomposition for all of these. However, for some other losses -- in particular the 0-1-loss for classification -- such a decomposition is proven to not exist (see \cite{todo}). In these cases, we can still consider the more general effect decompositions (see \ref{sec:todo}).

% TODO intuition plot for bregman divergence
% TODO examples for losses that are bregman divergences, see e.g. here:https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf


Bregman divergences have the one key property that their loss-effect terms collapse. That is, with bregman divergences, we have, for the the proper choice of $Z, Z'$:
$$
\LE{Z}{Z'} = \ell(Z, Z')
$$
This is given by the following two lemmas. 
% TODO motivate more on why centroids

% #TODO its not really harder, just need to understand Thm 0.1 of Pfau13 (and maybe mention that this is essentially a statement about effect terms, but wasnt explicitly considered)
% if we want to make the point about centroids etc clearer, we already give the basis for thm0.1, so might as well do that.
% #todo also provide thm0.1 (needed for bias it seems, but first check that they really are not the same)
% the following is required for variance and ambiguity
\begin{lemma} \label{thm:bregman-collapse-bias} (\cite{pfau}, Theorem 0.1 (b))
Let the generator $\phi: \mathcal{S} \to \mathbb{R}$ be a strictly convex, differentiable function. Let $Y$ be a random variable on $\mathcal{S}$. Then, for any $q \in \mathcal{S}$, it holds that
% #todo will need to define bregman centroid. what kind of centroid do i need for variance lemma?
$$
\Breg{y^\star}{q} = \mathbb{E}_{}\left[ \Breg{Y}{q}  - \Breg{Y}{y^\star} \right]
$$
if $y^\star$ is the right Bregman centroid.
\end{lemma}

This shows that bias-effect collapses to bias for Bregman divergences: 
$$\Breg{y^\star}{q^\star} = \mathbb{E}_{}\left[ \Breg{Y}{q^\star} - \Breg{Y}{y^\star} \right]$$

\begin{lemma} \label{thm:bregman-collapse-variances} (Generalised from \cite{ref:wood23})
Let $\vec y$ be a random vector. Let $\vec q_{Z}$ be a random vector dependent on another random variable $Z$. Then it holds that
% TODO not sure whether this is correct, seems to have the exact same shape as above...
$$
\Breg{q^\star}{q} = 
\mathbb{E}_{Y}\left[ \Breg{Y}{q} - \Breg{Y}{q^\star} \right]  \\[1em]
$$
if  $q^\star$ is the left Bregman centroid.
\end{lemma}

\begin{proof}
\begin{align*}
\mathbb{E}_{Y,Z}\left[ \Breg{y}{q} - \Breg{y}{q^\star} \right]  &= \dots
\end{align*}
\end{proof}
% TODO give proof here --relies only on linearity and "dual expectation"
% TODO note that this is dual expectation, and repeat lemmas from paper indicating that the dual expectations are indeed centroids
% TODO more on how this is all about variation around centroids and what the centroids are in example of variance and ambiguity
% TODO bias-variance-diversity decomp
% TODO highlight somewhere that this is a slightly different way of showing the decomp -- was previously shown by directly showing bias-variance decomp for it; here we start with effect decomp which trivially holds and then apply bregman collapse trick (which is inherent in the other proof too, but certainly not as clear)

% TODO write D, Theta in both equations, needs to be clearer I guess
This shows that variance- and ambiguity-effect collapse to variance and ambiguity. The variance (in the bias-variance sense) is the variance of the estimates $q_D$ dependent on $D$ with respect to different realisations of $D$ around its $D$-centroid. 
$$
\text{For $q^\star = \mathcal{E}_D\left[q_D\right]$:}\hspace{1em}
\Breg{q^\star}{q} = \mathbb{E}_{}\left[ \Breg{Y}{q^\star} - \Breg{Y}{q} \right]
$$
Ambiguity/Diversity is the variance of the estimates $q_\Theta$ dependent on $\Theta$ with respect to different realisations of $\Theta$ around its centroid.
$$
\text{For $\bar{q} = \mathcal{E}_\Theta\left[q_\Theta\right]$:}\hspace{1em}
\Breg{\bar{q}}{q} = \mathbb{E}_{}\left[ \Breg{Y}{\bar{q}} - \Breg{Y}{q} \right]
$$
% TOD do these have a geometric intuition?

% TODO also give bregman ambiguity decomp somewhere
% then mention that krogh showed this for KL-divergence and Geurts (as mentioned in Louppe p67) for squared error.

This yields a generalised bias-variance-diversity decomposition for bregman divergences as a special case of the corresponding effect decomposition \ref{todo}.

% TODO notation, (re)define symbols
% TODO clarify: expectation over X
% TODO clarify: q depends on X (and D?), q* depends only on X
% #todo writing E_Y|X here, reconcile with other things
% \begin{theorem} (\textit{Bias-variance-decomposition for Bregman divergences}) \label{thm:bregman-bias-variance-decomp}
% $$
% \mathbb{E}_{(X,Y), D}\left[ \Breg{Y}{q} \right] 
% = \underbrace{\mathbb{E}_{Y|X}\left[ \Breg{Y}{y^\star} \right]   }_{\text{noise}}
% + \underbrace{\Breg{y^\star}{q^\star} }_{\text{bias}}
% + \underbrace{\mathbb{E}_{D}\left[ \Breg{q^\star}{q} \right] }_{\text{variance}}
% $$
% % #todo make sure the order arguments is as requried by the lemmas
% \end{theorem}

\begin{theorem} (Bias-Variance-Diversity decomposition for Bregman divergences)
    % TODO define qstar and qbar in terms of (dual) expectations (left/right centroids)
\begin{align*}
\mathbb{E}_{(X,Y), D}\left[ \Breg{Y}{q} \right]  
&= \underbrace{\mathbb{E}_{Y|X}\left[ \Breg{Y}{y^\star} \right]   }_{\text{noise}} \\
&+
\underbrace{
    \frac{1}{M} \sum_{i=1}^M B_\phi\left(\overline{\mathbf{Y}}, \mathbf{q}_i^*\right)
}_{\overline{\text{bias}}} \\
&+
\underbrace{
    \frac{1}{M} \sum_{i=1}^M \mathbb{E}_D\left[B_\phi\left(\mathbf{q}_i^*, \mathbf{q}_i\right)\right]
}_{\overline{\text{variance}}} \\
&-
\underbrace{
    \mathbb{E}_D\left[\frac{1}{M} \sum_{i=1}^M B_\phi\left(\overline{\mathbf{q}}, \mathbf{q}_i\right)\right]
}_{\text{diversity}}
\end{align*}
\end{theorem}

% TODO describe terms
% TODO give a plot here how avg-bias, avg-variance do not change but diversity improves

% TODO note on how we require special ensemble combiner according to dual expectation for this to hold
% give table for combiners for squared-error KL-divergence etc


\section{Diversity is a measure of model fit}

"Sweet spot" of diversity. Review of previous experiments. Will need to go somewhere else.
% cf obsidian "sweet spot for diversity" and other notes


% TODO wood23 "MC-dropout"
\end{document} 