clc;
clear all;

num = 10;

p = 0.25;
k_max = 100;
epsilon = 1e-5;
t = 0.9;

err = zeros(num, 1);
rk = zeros(num, 1);
time = zeros(num, 1);
for i = 1:num
    
    m = i * 100;
    r = i;
    tau = i;
    ell = 5 * i;
    
    tic
    [A, B, e] = low_rank_recovery(m, m, p, r, tau, epsilon, ell, k_max);
    time(i) = toc;
    clear A;
    err(i) = e;
    rk(i) = true_rank(B, t);
    clear B;
end

figure();
subplot(1, 3, 1);
plot(100:100:num * 100, err, '-*');
xlabel('m = n');
ylabel('||error||');
subplot(1, 3, 2);
plot(100:100:num * 100, time, '-*');
xlabel('m = n');
ylabel('time (s)');
title(['p = ', num2str(p), ', epsilon = ', num2str(epsilon)]);
subplot(1, 3, 3);
plot(100:100:num * 100, rk, '-*');
xlabel('m = n');
ylabel('effective rank');