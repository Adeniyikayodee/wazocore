# Testing two AI models in your language

Thank you for helping. You will try two versions of an AI model on your own
laptop, **Model A** and **Model B**, and tell us which one works better in
Yoruba, Hausa or Igbo. You will not be told which is which. That is on purpose,
so your judgement stays fair. It takes about **45 minutes**, and nothing is sent
anywhere: both models run offline on your laptop.

## What you need

- A Mac (Apple M1 or newer) or a Windows laptop, with **at least 8 GB of
  memory**. More is better.
- **About 12 GB of free disk space.**
- Wi-Fi for the downloads (about 10 GB in total). After that, no internet is
  needed.

## 1. Install LM Studio (5 minutes)

1. Go to **lmstudio.ai**, download LM Studio for your computer and install it.
2. Open it once, then close it.

## 2. Download and add the two models (20 to 40 minutes, mostly waiting)

1. Download **model-A.gguf** and **model-B.gguf** from the link you were sent.
   Save both in your Downloads folder.
2. Open **Terminal** (Mac) or **PowerShell** (Windows).
3. Paste these three lines, pressing Enter after each.

**Mac:**

```
cd ~/Downloads
~/.lmstudio/bin/lms import -y -c --user-repo tester/model-A model-A.gguf
~/.lmstudio/bin/lms import -y -c --user-repo tester/model-B model-B.gguf
```

**Windows:**

```
cd $env:USERPROFILE\Downloads
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" import -y -c --user-repo tester/model-A model-A.gguf
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" import -y -c --user-repo tester/model-B model-B.gguf
```

4. Open LM Studio. Both models now appear under **My Models**.
5. If you are short of disk space, delete the two files from Downloads; LM
   Studio has kept its own copies.

## 3. Run the tasks (15 minutes)

1. In LM Studio, open **Chat**.
2. At the top, choose **model-A** and wait for it to load.
3. Start a **new chat** for each task in the task sheet, and paste the task
   exactly as written.
4. Write down how the answer reads, and roughly how long it took to finish.
5. Switch to **model-B** at the top and do the same five tasks again.

Leave LM Studio's settings as they are. Both models then run under the same
conditions.

**If a model will not load, or the laptop runs out of memory:** click the
model's name at the top, open its load settings, set **Context Length** to
4096, and load it again. Use the same value for both models.

Please use **the same language** for all five tasks. Choose the one you speak
best.

## 4. Tell us what you found (5 minutes)

Fill in the feedback form from the link you were sent. There are no right
answers. "They are the same" is a useful answer too.

*If anything goes wrong, send a screenshot to the person who invited you.*
