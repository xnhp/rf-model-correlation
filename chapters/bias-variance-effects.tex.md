%! suppress = LineBreak
\documentclass[../main.tex]{subfiles}
\begin{document}
%! language = Latex\begin{document}

    % TODO write somewhere that bayes error can also be considered a variance (see Adlam22 eq (5)

We begin this chapter by deriving the classical Bias-Variance-decomposition from first principles.  
...
Understanding the nature and the generality of bias- and variance-\textit{effect} will allow us to very naturally and simply develop a notion of ensemble diversity as a measure of model fit, just like bias and variance. 

% TODO here, or somewhere, story of [[sweet spot for diversity]] and review on previous work

\begin{marginfigure} \label{fig:bias-variance-tradeoff}
    \includegraphics[width=\textwidth]{bias-variance/bias_variance_decomposition_with_decision_trees.png}
    \caption{foo!}
    % TODO
\end{marginfigure}

% TODO recap of ensemble error as in preliminary "supervised learning"
% % TODO bias and variance are measures independent of particular data sample, can use to compare (on same data*set*)

\section{Bias, Variance and their Effects} \label{sec:bias-variance-effects}


\begin{marginfigure} \label{fig:variance-trees}
    \includegraphics[width=\textwidth]{bias-variance/test_error_of_individual_trees.png}
    \caption{
    Visualising the variance of \textcolor{blue}{Decision Tree} and \textcolor{orange}{Random Forest} models. Each glyph corresponds to the test error of one model trained on a random subset of the full available data. The variation of the test error around the mean test error across many dataset samples is exactly the variance.
    Not only do Random Forests show lower test errors on average, they seem to also have lower variance. We will explain this observation in \ref{todo}
    % TODO because, in fact, ensemble improvement is solely due to reduction in variance
    %   since var(qbar) = mean(var(qi)) - div/div-eff
    % i.e. the difference between mean tree variances and ensemble variance is exactly the ambiguity/diversity-effect
    }
\end{marginfigure}


\subsection{Deriving Bias- and Variance-Effect}

We will begin by considering the widely known bias-variance-decomposition for regression using the squared-error loss $\ell(y, y') \defeq (y- y')^2$. For sake of clarity, we will from now on take the dependence of $q_{D}(X)$ on $X$ as understood and write only $q_{D}$.

% TODO always hyphen, i.e. "squared-error" loss
The variance of $q_{D}$ with respect to the squared-error loss is commonly defined as the variance of the regressor around its expected value. This is the expected squared error between a random value ($q_{D}$) and the closest non-random value ($\mathbb{E}_{D}\left[q_{D}\right]$). 
$$
\Var{D}{q_{D}} := \mathbb{E}_{_{D}}\left[ (q_{D}- \mathbb{E}_{_{D}}\left[ q_{D} \right] )^2 \right]  = \min_{z} \mathbb{E}_{D}\left[ (q_{D} - z)^2 \right] 
$$
% TODO #todo already use colours here?
As such, $\mathbb{E}_{D}\left[ q_{D} \right]$ is a centroid to the different realisations of $q_{D}$ with respect to the loss function $ell(y,y') = (y-y')^2$.
This non-random centroid will turn out to be particularly interesting. For a random variable $Z$, let us denote its centroid with respect to $Y$ as
$$
Z^\star \defeq \arg \min_{z} \mathbb{E}_Y\left[ (Y - z)^2 \right]
$$

Let's consider two centroids: One with respect to the label distribution and one with respect to the model outputs.
\begin{itemize}
\item $y^\star(X) = \arg\min_{z} \mathbb{E}_{Y}\left[ (Y - z(X))^2 \right] = \mathbb{E}_{Y|X}\left[ Y \right]$ is the \textit{expected label} % TODO clarify
\item $q^\star(X) = \arg \min_{z} \mathbb{E}_{D}\left[ (q_{D}(X) - z(X))^2 \right]$ is the \textit{expected model}.
\end{itemize}

% As such, for the squared-error loss, we can write
% $$
% \Var{q_{D}} = \mathbb{E}_{D}\left[ L(q_{D}, q^\star) \right] 
% $$
Using this notation, the well-known bias-variance decomposition for the squared-error loss is given as follows.
Note that each variance term is the expected distance in terms of loss to a certain centroid.
\begin{align} \label{thm:sqerr-bias-variance-decomp}
\mathbb{E}_{(X,Y), D}\left[ (Y-q_{D}(X))^2 \right]  &= 
  \mathbb{E}_{(X,Y)}\left[ (Y-y^\star(X))^2 \right]  
+ \mathbb{E}_{X, D}\left[ (y^\star(X) - q_{D}(X))^2 \right]  \\
&= \underbrace{\mathbb{E}_{(X,Y)}\left[ (Y-y^\star(X))^2 \right]   }_{\Var{Y} ~~ \text{("noise")}}
+ \underbrace{\mathbb{E}_{X, D}\left[ (q^\star(X) - q_{D}(X))^2 \right]   }_{\Var{q} ~~ \text{("learner variance")}}
+ \underbrace{ \mathbb{E}_{X}\left[  (y^\star(X) - q^\star(X))^2\right] 
  }_{\BiasSq{Y, q} ~~ \text{("learner bias")}}
\end{align}
% #todo change BiasSq to just "Bias" -- square is only artifact of squared error
This decomposition is usually derived by expanding the square \cite{todo}. The cross-terms then vanish due to that $\mathbb{E}_{_{D}}\left[ q_{D} \right] = q^\star$ and $y^\star = \mathbb{E}_{Y|X}\left[ Y \right]$
This is but a special case of a more general structure applying to a certain class of losses. We will provide a more general proof later. % #todo ref

The first term, $\Var{}{Y}$ is independent of $D$ and $q_{D}$. This means we have no means of influencing it with our choice of $q_{D}$.
$\Var{}{Y}$ is also referred to as \textit{noise}, \textit{bayes error} or \textit{irreducible error}. 
The second term, $\Var{}{q}$ measures the variance of our model around its non-random centroid with respect to different realisations of the random training dataset $D$. This can be understood as a measure of spread of the learning algorithm with respect to different realisations of $D$.
The third term, $\BiasSq{q_{D}, Y}$ is the distance in terms of loss between the expected classifier and the expected label. This can be thought of as a measure of precision of our learning algorithm. 

Note that we developed two things:
One the one hand, we derived quantities that measure the notions of bias and variance. On the other hand, by virtue of these quantities appearing in the error decomposition \ref{thm:sqerr-bias-variance-decomp}, we have quantified the \textit{effect} these quantities have on the generalisation error.
This distinction is often overlooked because for many commonly used losses the quantities and their effects coincide. However, in general this is not necessarily true. We are particularly interested in the \zeroone-loss for classification and a decomposition of it that is independent of the target has been proven to not exist \cite{todo}. We will see that, while we cannot give a bias-variance decomposition for any loss, we can give a more general bias-variance-\textit{effect} decomposition for which the former is a special case.

The variance-effect is the expected change in loss caused by using $q_{D}$ instead of the non-random centroid $q^\star$. Likewise, the bias-effect is the expected change in loss caused by using the expected model instead of the expected label. 
\sidenote{In the original publication \cite{james_GeneralizationsBiasVariance_}, bias-effect is called the \textit{systematic effect}, i.e. the effect of the systematic components. However, it is clearer to call this \textit{bias-effect}, particularly when we begin to introduce notions of diversity in \ref{sec:measures-ensemble-diversity}.}
Formally:
\begin{align*}
\text{Variance-Effect} &\defeq \mathbb{E}_{D,Y}\left[ \ell(Y, q_{D}) - \ell(Y, q^\star) \right] \\
\text{Bias-Effect} &\defeq \mathbb{E}_{D,Y}\left[ \ell(Y, q^\star) - \ell(Y, y^\star) \right] 
\end{align*}

To shorten notation, and to give the equations a shape suggestive for later, we define:
\begin{definition} (Loss-Effect) For a loss functon $\ell$, and random variables $Y, Z, Z'$, we define the change in loss between $Z$ and $Z'$ in relation to $Y$ as:
  $$
  \LE{Z}{Z'} \defeq \ell(Y, Z) - \ell(Y, Z')
  $$
  \label{def:loss-effect}
\end{definition}

Putting this together, we can state a generalised bias-variance decomposition. The classical bias-variance decomposition for squared error (\ref{thm:sqerr-bias-variance-decomp}) is a special case of this (\ref{lemma:the-trick}).

% TODO widepar messes up spacing?
\begin{theorem} (Bias-Variance-Effect-Decomposition \cite{james_GeneralizationsBiasVariance_})
For any loss function $L$, it holds that
% TODO make ystar lowercase in other places aswell (really, it also is a function dependent on x (the bayes classifier?))
% TODO emphasize how this is really about changes in *loss*, directly comparing *predictions* only comes for bregman diverges -- see e.g. arguments on variance reduction
\begin{align*}
\mathbb{E}_{(X,Y), D}\left[ \ell(Y, q_{D}) \right]  
&= 
\underbrace{\mathbb{E}_{Y}\left[ \ell(Y, y^\star) \right]  }_{\text{noise}}
+ \underbrace{\mathbb{E}_{(X,Y)}\left[ \LE{q^\star, y^\star} \right] }_{\text{bias-effect}}
+ \underbrace{\mathbb{E}_{(D,Y)}\left[ \LE{q_D, q^\star} \right] }_{\text{variance-effect}}  \\
\\ 
\text{for} \hspace{1em}
& \LE{q^\star}{y^\star} = \ell(Y, q^\star) - \ell(Y, y^\star) \\ \\
&\LE{q_{D}}{q^\star} = \ell(Y, q_{D}) - \ell(Y, q^\star)
\end{align*}
\end{theorem}
This decomposition holds for \textit{any} loss function $\ell$ since, by linearity of expectation, the individual terms on the right-hand-side simply cancel out. Further, this decomposition is independent of the definitions of $y^\star$ and $q^\star$.

\end{document}