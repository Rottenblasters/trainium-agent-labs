# init_trainium.ps1

Write-Host "=== Setting AWS credentials ==="

# Load credentials into THIS PowerShell session
. .\set_aws_credentials.ps1

Write-Host "=== Checking AWS identity ==="

aws sts get-caller-identity

if ($LASTEXITCODE -ne 0) {
    Write-Error "AWS authentication failed. Check/refresh your credentials."
    exit 1
}

Write-Host "Pointing Kubectl at the EKS cluster..."

aws eks update-kubeconfig --name hack-hyd --region ap-south-2

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to configure EKS."
    exit 1
}

Write-Host "Finding Pod 258"

kubectl get pod seat-258

if ($LASTEXITCODE -ne 0) {
    Write-Error "kubectl could not find Pod 258."
    exit 1
}

Write-Host "Get a Shell ready on Pod 258"

kubectl exec -it seat-258 -- bash

Write-Host "=== Trainium environment ready ==="