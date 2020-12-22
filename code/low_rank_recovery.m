function [B, B_hat, error] = low_rank_recovery(m, n, p, r, tau, epsilon, ell, k_max)

if nargin == 7
    k_max = 100;
end
if nargin == 6
    ell = 3 * r;
    k_max = 100;
end
if nargin == 5
    epsilon = 1e-8;
    ell = 3 * r;
    k_max = 100;
end
if nargin == 4
    tau = r;
    epsilon = 1e-8;
    ell = 3 * r;
    k_max = 100;
end
if nargin == 3
    r = round(m / 100);
    tau = r;
    epsilon = 1e-8;
    ell = 3 * r;
    k_max = 100;
end
if nargin == 2
    p = 0.1;
    r = round(m / 100);
    tau = r;
    epsilon = 1e-8;
    ell = 3 * r;
    k_max = 100;
end
if nargin == 1
    n = m;
    p = 0.1;
    r = round(m / 100);
    tau = r;
    epsilon = 1e-8;
    ell = 3 * r;
    k_max = 100;
end
if nargin == 0
    error('Input an argument.');
end

U0 = randn(m, r);
V0 = randn(n, r);
N = randn(m, n);

B = U0 * V0' + 0.1 * N;
Omega = rand(m, n) < p;

R = Omega .* B;
Q = 0;

k = 0;
while k<k_max
    
    k = k + 1;
    Z = Omega .* R;
    [u, sigma, v] = svds(Z, 1);
    Delta = Omega .* (tau * u * v') - Q;
    rho = trace(Delta' * R);
    
    if rho < epsilon
        break;
    end
    
    theta = min(1, rho / norm(Delta, 'fro')^2);
    R = R - theta .* Delta;
    Q = Q + theta .* Delta;
    
end

[U, Sigma, V] = svds(Z, ell);

%Recovery of the primal%
ind = find(Omega);

B_col = reshape(B, m * n, 1);
G = zeros(m * n, ell);
for i = 1:ell
    G(:, i) = reshape(U(:, i) * V(:, i)', m * n, 1);
end

G = G(ind, :);
B_col = B_col(ind);

S = diag(lsqnonneg(G, B_col));
B_hat = U * S * V';

error = norm(B - B_hat, 'fro');

end

