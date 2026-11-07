%% rig_inputs.m  Excitation inputs for the observer recordings on the Feedback digital pendulum
%  (mechanical unit 33-200, controller 33-201).
%
% This is the code that ran on 6 Nov 2026 on the lab PC (Windows XP, MATLAB R2009a). It avoids
% implicit expansion (bsxfun in the multisine line), which R2009a lacks.
%
% Run this once in MATLAB before the session. It creates one [time, volts] array per run in the
% workspace. Feed the array of the current run into the From Workspace block that drives the
% motor DAC (Data = u_run4 etc.; "Form output after final data value by" = "Setting to zero").
%
% Every run is recorded WITH the cart-centring loop switched on. The CSV column command_v must be
% the total voltage sent to the DAC (excitation plus centring loop).
%
% If you change A_twotone or A_heldout below, save and run this script again: the arrays are
% computed from them and do not update by themselves.

Ts = 0.001;                 % 1 kHz, the Simulink fixed step

A_twotone = 0.25;           % V   run 4 and run 7. On 6 Nov 2026: 0.25 for run4 trial 1, then 0.35 for
                            %     run4 trials 2-3 and run7, set at the prompt (see below). 0.40 was not used:
                            %     the real cart travels about 1.5x more than the simulation predicted.
A_fading  = 0.40;           % V   run 5
A_heldout = 0.70;           % V   run 6; at least 1.5x the break-away voltage found in run 3

t_of = @(T) (0:Ts:T)';
ramp = @(t, tr) 0.5 * (1 - cos(pi * min(t / tr, 1)));      % smooth 0 -> 1 start

% Run 1: rest, 20 s
t = t_of(20);  u_run1 = [t, zeros(size(t))];

% Run 2: free swing, 60 s (lift the pendulum 20-30 deg, release it after pressing Run)
t = t_of(60);  u_run2 = [t, zeros(size(t))];

% Run 3: break-away staircase, 45 s. Every 3 s: +a for 0.5 s, rest 1 s, -a for 0.5 s, rest 1 s,
% with a = 0.10, 0.15, ..., 0.80 V. The first level that moves the cart is the break-away voltage.
t = t_of(45);  k = floor(t / 3);  ph = t - 3 * k;  a = 0.10 + 0.05 * k;
u = a .* ((ph < 0.5) - (ph >= 1.5 & ph < 2.0));
u(a > 0.80 + 1e-9) = 0;
u_run3 = [t, u];

% Run 4: two-tone (shape of scenario S1), 50 s
t = t_of(50);  u_run4 = [t, A_twotone * (2 * sin(1.5 * t) + 1.2 * cos(3 * t))];

% Run 5: fading, non-PE (shape of scenario S3), 60 s
t = t_of(60);  u_run5 = [t, A_fading * 3 * exp(-t / 8) .* sin(1.6 * t)];

% Run 6: held-out inputs H1-H4 of the shared benchmark, 12 s each, peak A_heldout
t = t_of(12);
% H1 filtered square: +-1, period 4 s, through a first-order lag tau = 0.3 s
k = floor(t / 2);  level = 1 - 2 * mod(k, 2);  prev = -level;  prev(k == 0) = 0;
u_run6_square = [t, A_heldout * (level + (prev - level) .* exp(-(t - 2 * k) / 0.3))];
% H2 multisine: 0.7, 2.2, 3.7, 5.3 rad/s with the benchmark's fixed phases, peak 1
w  = [0.7, 2.2, 3.7, 5.3];  phs = [1.124281, 4.020693, 2.935934, 2.327923];
ms = ramp(t, 1) .* 0.35 .* sum(sin(bsxfun(@plus, t * w, phs)), 2);
u_run6_multisine = [t, A_heldout * ms / max(abs(ms))];
% H3 slow chirp 0.1 -> 0.6 Hz
u_run6_chirp = [t, A_heldout * ramp(t, 1) .* sin(2 * pi * (0.1 * t + 0.5 * 0.5 * t.^2 / 12))];
% H4 pulse train: +1 for 0.3 s at 0.2 s, -1 for 0.3 s at 1.2 s, period 2 s
ph = mod(t, 2);
u_run6_pulse = [t, A_heldout * ((ph >= 0.2 & ph < 0.5) - (ph >= 1.2 & ph < 1.5))];

% Run 7: two-tone again with the changed pendulum (extra mass or alternate rod), 50 s
u_run7 = u_run4;

% On 6 Nov 2026 the 0.35 V two-tone was created at the prompt after this script, with
%   A_twotone = 0.35; t = (0:0.001:50)'; u_run4 = [t, A_twotone*(2*sin(1.5*t) + 1.2*cos(3*t))];
%   u_run7 = u_run4;
% From Workspace data are compiled into the real-time module: rebuild (Ctrl+B) after every change.

fprintf('Inputs ready: u_run1 ... u_run7 (u_run6_square, _multisine, _chirp, _pulse).\n');
fprintf('Peak volts: run3 %.2f, run4 %.2f, run5 %.2f, run6 %.2f\n', ...
        max(abs(u_run3(:, 2))), max(abs(u_run4(:, 2))), max(abs(u_run5(:, 2))), A_heldout);
