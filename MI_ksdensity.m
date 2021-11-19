function [MI, MInormalized] = MI_ksdensity(X, y)
%
% General call: [MI, MInormalized] = MI_ksdensity(X, y)
%
%% Function to get the mutual information using kernel density estimation
%  this is a basic computation of mutual information relying on kernel density estimation, but I wanted to keep things simple in this implementation 
%
% Inputs:  X       -> N by M matrix, N = observations, M = features
%          y       -> N by 1 vector with the numerical outputs
%__________________________________________________________________________
% optional inputs:  
% =========================================================================
% Outputs: MI      -> Mutual Information (computed using Parzen windows)--- each MI entry has the mutual information between the X(i) and y
%                     (the MI units in this implementation are 'Nats') 
%     MInormalized -> Normalized Mutual Information values with respect to
%                     the mutual information of y. This enables direct
%                     comparison between the association strength of the
%                     features in X with respect to y
% =========================================================================
%
% Part of the "Statistical Machine Learning" Toolbox by A. Tsanas
%
% -----------------------------------------------------------------------
% Useful references:
% 
% 1) T.M. Cover and J.A. Thomas: Elements of information theory,
%    Wiley-interscience, 2nd edition, 2006 
% -----------------------------------------------------------------------
%
% Last modified on 15 February 2014
%
% Copyright (c) Athanasios Tsanas, 2014
%
% ********************************************************************
% If you use this program please cite:
%
%    A. Tsanas: Accurate telemonitoring of Parkinson's disease symptom
%    severity using nonlinear speech signal processing and statistical 
%    machine learning, D.Phil. thesis, University of Oxford, UK, 2012
% OR
%    A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig: "Nonlinear speech 
%    analysis algorithms mapped to a standard metric achieve clinically 
%    useful quantification of average Parkinson’s disease symptom severity",
%    Journal of the Royal Society Interface, Vol. 8, pp. 842-855, June 2011 
% ********************************************************************
%
% For any question, to report bugs, or just to say this was useful, email
% tsanasthanasis@gmail.com

[N,M]=size(X);
x2=y; % notational convenience

for i=1:M
    x1 = X(:,i);

    %% Compute the pdfs
    [p1, xc1, bw1] = ksdensity(x1); % marginal density estimate x1
    [p2, xc2, bw2] = ksdensity(x2); % marginal density estimate x2
    p12 = ksdensity2d(x1, x2, [bw1, bw2], xc1, xc2); % joint density estimate
    % Normalize the probalities calling the norm01 function 
    p1n = norm01(p1);
    p2n = norm01(p2);
    p12n = norm01(p12);

    %% Compute the MI    
   [NN,MM] = size(p12);
    
    % simple application of the general MI computation equation
    MI_norm_pdf=0;
    for ii=1:NN
        for j=1:MM
            if (p12n(ii,j)>0 && p1n(ii)>0 && p2n(j)>0)
                MI_norm_pdf = MI_norm_pdf + p12n(ii,j) * log(p12n(ii,j)/(p1n(ii)*p2n(j)));        
            end       
        end
    end
    MI(i) = MI_norm_pdf;
end

% Now get the normalized version of MI with respect to the response
% variable, allowing direct comparison of MI values amongst the features in
% the design matrix X
if(nargout>1) % do the computation only if requested by the user
    norm_factor = MI_ksdensity(y, y); % normalization factor: the MI of the response with itself
    MInormalized = MI/norm_factor;
end

end % end of main function

% ========================================================================
%% additional functions

function p12 = ksdensity2d(x1, x2, bw, gridx1, gridx2)
% function for the computation of 2-dimensional kernel density estimation

x1 = x1(:);
x2 = x2(:);
N = size(x1,1);

m1 = length(gridx1);
m2 = length(gridx2);

%% Main part of the computations

% Compute the kernel density estimate
[gridx2,gridx1] = meshgrid(gridx2,gridx1);
X1 = repmat(gridx1, [1,1,N]);
X2 = repmat(gridx2, [1,1,N]);
mu1(1,1,:) = x1; mu1 = repmat(mu1,[m1,m2,1]);
mu2(1,1,:) = x2; mu2 = repmat(mu2,[m1,m2,1]);
p12 = sum(normpdf(X1,mu1,bw(1)).*normpdf(X2,mu2,bw(2)), 3)/N;

end

function p_new = norm01(p_old)
% Function that normalizes the sum of the probabilities between 0 and 1

[N, M] = size (p_old);

if (M>1) % check if input is matrix
    for i=1:N
        for j=1:M
            if (p_old(i,j)<0) 
                p_old(i,j)=0; % getting negative entries to zero
            end
        end
    end
    sump_old=sum(sum(p_old));
    p_new=p_old./sump_old; %normalize all entries so that their sum=1

else %input is vector
    for i=1:N
        if (p_old(i)<0) 
            p_old(i)=0; % getting negative entries to zero
        end
    end
    sump_old=sum(p_old);
    p_new=p_old./sump_old; %normalize all entries so that their sum=1
end

end