# Test deployment on a small cloud VM

One small VM runs the whole stack: LibreChat, MongoDB, Meilisearch and the agent core.
The walkthrough below uses **Azure**, a spot `Standard_B2s` (2 vCPU, 4 GB) in **Central India**,
and the portal's Cloud Shell, so nothing needs installing locally. Google Cloud commands for the
same setup are at the end.

**This is a test setup.** It serves plain HTTP, so passwords and chats travel unencrypted.
Use a throwaway password, turn registration off once your account exists, and keep real data out
of it.

---

## 1. Create the VM (Azure)

Sign in at [portal.azure.com](https://portal.azure.com) and open **Cloud Shell** (the `>_` icon in
the top bar). Choose **Bash**; the first run offers to create a small storage account for your home
directory, or an ephemeral session, either is fine.

```bash
az group create --name hr-demo-rg --location centralindia

az vm create \
  --resource-group hr-demo-rg \
  --name hr-demo \
  --image Ubuntu2404 \
  --size Standard_B2s \
  --priority Spot --eviction-policy Deallocate --max-price -1 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard \
  --os-disk-size-gb 32

az vm open-port --resource-group hr-demo-rg --name hr-demo --port 3080 --priority 1001
az vm show -d -g hr-demo-rg -n hr-demo --query publicIps -o tsv
```

That last line prints the public IP. Only tcp:3080 is opened; leave 8088 closed, because the agent
console can browse and reset the demo database. `docker-compose.yml` binds it to localhost anyway.

If anything fails:

- *Unrecognised image alias* — use `--image Canonical:ubuntu-24_04-lts:server:latest`, or `Ubuntu2204`.
- *SkuNotAvailable / no spot capacity* — drop the three spot flags for a regular VM, or try
  `--location southindia` / `--size Standard_B2als_v2`.
- *Quota errors on a free trial* — free subscriptions often have no spot quota. Same fix: drop the
  spot flags.

## 2. Get the code onto the VM

```bash
ssh azureuser@YOUR_VM_IP
sudo apt-get update && sudo apt-get install -y git
```

For a **private** repo, give the VM a read-only deploy key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "hr-demo-vm"
cat ~/.ssh/id_ed25519.pub
```

Paste that line into the repo on GitHub under **Settings → Deploy keys → Add deploy key**, leaving
"Allow write access" unticked. Then, still on the VM:

```bash
git clone git@github.com:YOUR_USER/YOUR_REPO.git agenticsys
cd agenticsys
```

(A public repo just needs the `https://` URL and no key.)

## 3. Start everything

```bash
GEMINI_API_KEY=AIza_your_key \
AGENT_DEFAULT_MODEL=gemini/gemini-3.1-flash-lite \
bash deploy/vm-bootstrap.sh
```

The script installs Docker, writes `.env` (generating LibreChat's secrets), points
`DOMAIN_CLIENT` / `DOMAIN_SERVER` at the VM's public IP, builds the images, starts the containers
and runs the smoke test. First run takes several minutes.

Re-running it is safe, and it is the fix if the public IP ever changes.

## 4. Lock it down

Open `http://YOUR_VM_IP:3080`, register your account (the first one is admin), then:

```bash
sed -i 's/^ALLOW_REGISTRATION=.*/ALLOW_REGISTRATION=false/' .env
docker compose up -d
```

Without this, anyone who finds the IP can sign up and spend your model quota.

## 5. Day to day

```bash
docker compose logs -f librechat          # or agent-core
docker compose up -d                      # apply .env changes
docker compose down                       # stop the stack, keep the data
```

The admin console stays closed to the internet. Reach it from your own machine with an SSH tunnel,
then open `http://localhost:8088`:

```bash
ssh -L 8088:localhost:8088 azureuser@YOUR_VM_IP
```

Cost control, and the one Azure trap worth knowing: **"Stop" in the portal still bills for the VM
unless it is deallocated.** Use:

```bash
az vm deallocate -g hr-demo-rg -n hr-demo     # stops the charges (disk still costs a little)
az vm start      -g hr-demo-rg -n hr-demo     # bring it back
az group delete  -n hr-demo-rg --yes          # delete everything when you are done
```

A Standard SKU public IP keeps its address across a deallocate/start cycle, so the URL stays put.
A spot VM can be evicted at any time, which deallocates it; starting it again is the whole recovery.

## Sizing notes

| | |
|---|---|
| `Standard_B2s` (2 vCPU, 4 GB) | Comfortable. LibreChat ~1 GB, MongoDB ~0.5 GB, Meilisearch ~0.3 GB, agent core ~0.5 GB. |
| `Standard_B2als_v2` (2 vCPU, 4 GB) | Newer AMD burstable, usually a bit cheaper. Same fit. |
| 2 GB sizes (`B1ms`) | Only with `SEARCH=false` and the meilisearch container removed. Builds are slow. |
| Disk | 32 GB. The images come to roughly 4 GB. |
| Local models | Not on this VM. Ollama needs far more RAM and ideally a GPU; point `OLLAMA_BASE_URL` at a machine that has one. |

---

## The same thing on Google Cloud

In [Cloud Shell](https://console.cloud.google.com):

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com

gcloud compute instances create hr-demo \
  --zone=asia-south1-a \
  --machine-type=e2-medium \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=20GB --boot-disk-type=pd-balanced \
  --tags=hr-demo

gcloud compute firewall-rules create hr-demo-ui \
  --allow=tcp:3080 --target-tags=hr-demo --source-ranges=0.0.0.0/0

gcloud compute ssh hr-demo --zone=asia-south1-a
```

Steps 2 to 4 are identical. For the console tunnel use
`gcloud compute ssh hr-demo --zone=asia-south1-a -- -L 8088:localhost:8088`, and to stop paying,
`gcloud compute instances stop hr-demo --zone=asia-south1-a`. Google's external IPs are ephemeral
by default and change on stop/start, so re-run the bootstrap script afterwards.
