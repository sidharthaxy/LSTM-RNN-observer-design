function export_run(name, t, X, cols)
%EXPORT_RUN  Write one rig recording as the CSV that experiments/run_rig_evaluation.py reads.
%
%   export_run('run4_twotone_trial1', t, X, [cart angle command])
%
%   t     time vector of the logged run [s]
%   X     logged signals, one column per signal (the To Workspace / simout data)
%   cols  column numbers in X of [cart position, pendulum angle, total DAC command in volts]
%
%   With a To Workspace block saved as "Structure With Time":
%       export_run('run4_twotone_trial1', simout.time, simout.signals.values, [3 1 5])
%   With "Array" format and time saved as tout:
%       export_run('run4_twotone_trial1', tout, simout, [3 1 5])
%
%   Check the column numbers once on the Scope: push the cart by hand and see which column moves.
%   The file name must contain the run keyword (rest, freeswing, deadzone, twotone, fading,
%   heldout, mass_step); see README §12.1.

t = t(:);
T = table(t, X(:, cols(1)), X(:, cols(2)), X(:, cols(3)), ...
          'VariableNames', {'time_s', 'cart_pos', 'pend_angle', 'command_v'});
writetable(T, [name '.csv']);

dt = diff(t);
fprintf('%s.csv: %d rows, %.3f s, median step %.3f ms, largest step %.3f ms\n', ...
        name, height(T), t(end) - t(1), 1e3 * median(dt), 1e3 * max(dt));
fprintf('  cart range [%g, %g], angle range [%g, %g], command range [%g, %g]\n', ...
        min(T.cart_pos), max(T.cart_pos), min(T.pend_angle), max(T.pend_angle), ...
        min(T.command_v), max(T.command_v));
if max(dt) > 1.5 * median(dt)
    warning('Samples were dropped (gaps in time_s). Repeat the run if the gap is long.');
end
end
