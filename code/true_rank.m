function r = true_rank(A, t)

s = svd(A);

for i = 1:size(s)
    if norm(s(1:i)) >= t * norm(s)
        r = i;
        break;
    end
end

end

