function export_run(name, t, X, cols)
%EXPORT_RUN  Write one rig recording as the CSV that experiments/run_rig_evaluation.py reads.
%
%   This is the code that ran on 6 Nov 2026 on the lab PC (Windows XP, MATLAB R2009a), which has no
%   table/writetable; the CSV is written with fopen/fprintf.
%
%   export_run('run4_twotone_trial1', t, X, [cart angle command])
%
%   t     time vector of the logged run [s]
%   X     logged signals, one column per signal (the To Workspace / simout data)
%   cols  column numbers in X of [cart position, pendulum angle, total DAC command in volts]
%
%   With a To Workspace block saved as "Structure With Time" (the call used for every recording on
%   6 Nov 2026, with Mux1 inputs 1 position, 2 angle, 3 total command):
%       export_run('run4_twotone_trial1', simout.time, simout.signals.values, [1 2 3])
%   With "Array" format and time saved as tout:
%       export_run('run4_twotone_trial1', tout, simout, [1 2 3])
%
%   Check the column numbers once: push the cart by hand and see which column moves.
%   The file name must contain the run keyword (rest, freeswing, deadzone, twotone, fading,
%   heldout, mass_step). The CSV is written to MATLAB's current folder.

t = t(:);
D = [t, X(:, cols(1)), X(:, cols(2)), X(:, cols(3))];

fname = [name '.csv'];
fid = fopen(fname, 'w');
if fid < 0
    error('Could not open %s for writing. Check the current folder is writable.', fname);
end
fprintf(fid, 'time_s,cart_pos,pend_angle,command_v\n');
fprintf(fid, '%.6f,%.9g,%.9g,%.9g\n', D');
fclose(fid);

dt = diff(t);
fprintf('%s: %d rows, %.3f s, median step %.3f ms, largest step %.3f ms\n', ...
        fname, size(D, 1), t(end) - t(1), 1e3 * median(dt), 1e3 * max(dt));
fprintf('  cart range [%g, %g], angle range [%g, %g], command range [%g, %g]\n', ...
        min(D(:, 2)), max(D(:, 2)), min(D(:, 3)), max(D(:, 3)), min(D(:, 4)), max(D(:, 4)));
if max(dt) > 1.5 * median(dt)
    warning('Samples were dropped (gaps in time_s). Repeat the run if the gap is long.');
end
end
