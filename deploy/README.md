# Test deployment on a small cloud VM

A single small VM runs the whole stack: LibreChat, MongoDB, Meilisearch and the agent core.
This walkthrough uses Google Cloud, a spot `e2-medium` (2 vCPU, 4 GB) in `asia-south1` (Mumbai),
and Google's browser-based Cloud Shell, so nothing has to be installed locally.

**This is a test setup.** It serves plain HTTP, so passwords and chats travel unencrypted.
Use a throwaway password, turn registration off once your account exists, and don't put
anything real in it.

---

## 1. Create the VM

In the [Cloud console](https://console.cloud.google.com), create a project and make sure billing
is on (new accounts come with free credit). Open **Cloud Shell** (the `>_` icon, top right) and run:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com

gcloud compute instances create hr-demo \
  --zone=asia-south1-a \
  --machine-type=e2-medium \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=20GB --boot-disk-type=pd-balanced \
  --tags=hr-demo

gcloud compute firewall-rules create hr-demo-ui \
  --allow=tcp:3080 --target-tags=hr-demo --source-ranges=0.0.0.0/0 \
  --description="LibreChat UI (test)"
```

Only tcp:3080 is opened. Leave 8088 closed: the agent console can browse and reset the demo
database, and `docker-compose.yml` binds it to localhost for that reason.

Spot VMs are 60-70% cheaper and can be reclaimed at any time, in which case the VM stops.
Start it again with `gcloud compute instances start hr-demo --zone=asia-south1-a`.

## 2. Get the code onto the VM

```bash
gcloud compute ssh hr-demo --zone=asia-south1-a
```

For a **private** repo, give the VM a read-only deploy key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "hr-demo-vm"
cat ~/.ssh/id_ed25519.pub
```

Copy that line into the repo on GitHub: **Settings → Deploy keys → Add deploy key**, paste, leave
"Allow write access" unticked. Then, still on the VM:

```bash
git clone git@github.com:YOUR_USER/YOUR_REPO.git agenticsys
cd agenticsys
```

(A public repo just needs the `https://` clone URL and no key.)

## 3. Start everything

```bash
GEMINI_API_KEY=AIza_your_key \
AGENT_DEFAULT_MODEL=gemini/gemini-3.1-flash-lite \
bash deploy/vm-bootstrap.sh
```

The script installs Docker, writes `.env` (generating LibreChat's secrets), points
`DOMAIN_CLIENT` / `DOMAIN_SERVER` at the VM's external IP, builds the images, starts the
containers and runs the smoke test. First run takes a few minutes.

Re-running it is safe, and it is the fix for a changed IP after a stop/start.

## 4. Lock it down

Open `http://VM_IP:3080`, register your account (the first one is admin), then:

```bash
sed -i 's/^ALLOW_REGISTRATION=.*/ALLOW_REGISTRATION=false/' .env
docker compose up -d
```

Without this, anyone who finds the IP can sign up and spend your model quota.

## 5. Day to day

```bash
docker compose logs -f librechat          # or agent-core
docker compose up -d                      # apply .env changes
docker compose down                       # stop (keeps data)

# Admin console, without exposing it: run this on your own machine, then open localhost:8088
gcloud compute ssh hr-demo --zone=asia-south1-a -- -L 8088:localhost:8088
```

Costs accrue while the VM runs. Stop it when you're done for the day
(`gcloud compute instances stop hr-demo --zone=asia-south1-a`); you keep paying only for the
20 GB disk. The external IP is ephemeral and changes on stop/start, so re-run the bootstrap
script afterwards, or reserve a static IP if that gets annoying.

## Sizing notes

| | |
|---|---|
| `e2-medium` (4 GB) | Comfortable. LibreChat ~1 GB, MongoDB ~0.5 GB, Meilisearch ~0.3 GB, agent core ~0.5 GB. |
| `e2-small` (2 GB) | Works if you set `SEARCH=false` in `.env` and drop the meilisearch container. Builds are slow. |
| Disk | 20 GB. The images come to roughly 4 GB. |
| Local models | Not on this VM. Ollama needs far more RAM (and ideally a GPU); point `OLLAMA_BASE_URL` at a machine that has one. |
