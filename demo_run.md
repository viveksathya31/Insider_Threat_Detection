# Live Demo Run-of-Show (5-10 min)

## Before your guides arrive (do NOT do this live -- too slow / risk of failure)

1. Make sure everything already runs cleanly once, so you're not debugging live:
   ```
   python scripts/evaluate_test.py
   python scripts/demo_single_day.py --list
   ```
2. Pick and test your demo day now (see step 3 below) so you know what it'll show.
3. Open two terminal tabs: one in the project root, one ready for the demo script.
4. Increase your terminal font size. Guides need to read this from across a table.
5. Close anything else that might pop up a notification during the demo.

---

## Live script (~7 min, trim the optional parts if you're tighter on time)

### 1. One-line framing (30 sec, spoken, no terminal)
"We're detecting insider threats from enterprise activity logs -- who's logging in,
copying files, browsing, emailing -- using a graph neural network instead of traditional
rule-based or tabular ML. Today I'll show the working pipeline and real evaluation
numbers on data the model has never seen."

### 2. Show the data scale (30 sec)
```
ls -la data/raw/r4.2/
```
Say: "16 gigabytes of real enterprise logs, half a million user-days, we're detecting
insider threats which normally show up in about 1 in 500 user-days."

### 3. Show the final, honest result (2-3 min -- this is the main event)
```
python scripts/evaluate_test.py
```
While it runs (~10-20 sec), say: "This is evaluating on the test period -- February
through May 2011 -- which the model has never trained on or been tuned against."

When it finishes, point at:
- `test PR-AUC: 0.2557 ... 175.2x better than random baseline`
  → "175 times better than chance, on genuinely unseen time."
- The confusion matrix line: `of 143 truly malicious user-days, caught 67`
  → "Catches about half of malicious activity, with roughly 2-3 false alarms per
     real catch -- which is realistic for a triage tool an analyst reviews, not
     something meant to act autonomously."

### 4. Concrete example -- one real day (2-3 min)
```
python scripts/demo_single_day.py
```
Say while it runs: "Instead of just the aggregate number, here's the model actually
working on one specific day from the test period."

Point at:
- The ranked list of highest-risk users -- "these are the model's top suspects for
  that day, purely from graph structure and behavior, no user-identity shortcuts."
- The "Caught" / "Missed" lines -- be honest about both. If it caught someone, say so.
  If it missed someone, that's fine to show too -- "this is exactly the kind of case
  we're working on improving next."

### 5. Close with what's next (30-60 sec, spoken)
"Two known issues we're actively working on: the model overfits after about epoch 18,
so we're adding regularization next. And it currently looks at each day in isolation --
adding a temporal layer so it remembers a user's recent history is the next big
architectural step, along with explainability so an analyst can see *why* a user was
flagged."

---

## If something goes wrong live

- **A command hangs or errors:** don't debug live. Say "let me show you the output from
  my last run instead" and have a terminal scrollback or a saved text file of the last
  successful run ready as backup.
- **Guides ask a number you don't have memorized:** it's fine to say "let me pull that up"
  and actually pull it up -- these are real scripts, not slides, so checking live is
  expected and looks credible, not unprepared.
- **Guides want to see the code, not just output:** have `src/model.py` open in an editor
  tab ready to switch to, don't scramble to find it.

## Backup: save a known-good output in case of live failure

Right before the demo, run this once and save the output as a fallback you can paste in
if live execution fails for any reason (network blip, MPS hiccup, etc.):
```
python scripts/evaluate_test.py > /tmp/eval_backup.txt 2>&1
python scripts/demo_single_day.py > /tmp/demo_backup.txt 2>&1
cat /tmp/eval_backup.txt /tmp/demo_backup.txt
```