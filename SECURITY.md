# Safety, privacy, and responsible handling

This is an **experimental** EEG communication prototype, not a medical or clinical product.

- **Visual stimulus safety:** The SSVEP interface uses flickering visuals. Do not expose an audience to flashing targets without screening, consent, a mentor-approved setup, and an immediate stop mechanism. Follow g.tec instructions and venue rules.
- **Participant privacy:** Treat raw EEG, recordings, timestamps linked to participants, and names as sensitive. Keep local data out of GitHub. Publish only consented postcards and minimal metadata.
- **Accuracy/intent:** Distinguish raw decoder predictions, AI suggestions, and user-approved messages. Do not silently auto-send suspected intent.
- **Credentials:** Store API keys in local environment variables or secret managers. Do not commit `.env` or tokens, including in screenshots and logs.
- **Vulnerability reporting:** Tell the team maintainer privately rather than opening an issue that exposes private data or credentials.
