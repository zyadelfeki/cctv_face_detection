# Kubernetes Deployment Guide

## 🚀 Production-Ready Kubernetes Deployment

This directory contains enterprise-grade Kubernetes manifests for deploying the CCTV Face Detection system with maximum security and reliability.

## 📋 Prerequisites

### Required Tools
- Kubernetes cluster (v1.24+)
- kubectl (v1.24+)
- helm (v3.0+)
- cert-manager (for TLS)
- External Secrets Operator (recommended)
- ingress-nginx controller

### Infrastructure Requirements
- **Minimum**: 3 nodes, 8 vCPUs, 16GB RAM total
- **Recommended**: 5+ nodes, 20+ vCPUs, 40GB+ RAM
- **Storage**: SSD-backed persistent volumes
- **Network**: CNI with NetworkPolicy support (Calico, Cilium, etc.)

## 🔒 Security Features

✅ **Pod Security Standards**: Restricted profile enforced  
✅ **Network Policies**: Zero-trust networking (default deny)  
✅ **RBAC**: Minimal permissions per component  
✅ **Non-root Containers**: All containers run as uid 10000  
✅ **Read-only Filesystems**: Where applicable  
✅ **Resource Limits**: CPU and memory quotas  
✅ **TLS 1.3**: Enforced for all external traffic  
✅ **Security Headers**: HSTS, CSP, X-Frame-Options, etc.  
✅ **Secret Management**: External Secrets Operator integration  
✅ **WAF**: ModSecurity with OWASP CRS  

## 📦 Deployment Steps

### Step 1: Create Namespace

```bash
kubectl apply -f namespace.yaml
```

### Step 2: Create Secrets

**Option A: Manual (Development)**

```bash
# Generate secure passwords
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export SECRET_KEY=$(openssl rand -hex 32)
export ENCRYPTION_KEY=$(openssl rand -hex 32)

# Create secrets
kubectl create secret generic postgres-secret \
  --from-literal=username=cctv_user \
  --from-literal=password=${POSTGRES_PASSWORD} \
  -n cctv-detection

kubectl create secret generic redis-secret \
  --from-literal=password=${REDIS_PASSWORD} \
  -n cctv-detection

kubectl create secret generic cctv-secrets \
  --from-literal=database-url="postgresql://cctv_user:${POSTGRES_PASSWORD}@postgres:5432/cctv_detection" \
  --from-literal=redis-url="redis://:${REDIS_PASSWORD}@redis:6379/0" \
  --from-literal=secret-key=${SECRET_KEY} \
  --from-literal=encryption-key=${ENCRYPTION_KEY} \
  -n cctv-detection
```

**Option B: External Secrets Operator (Production)**

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace

# Apply External Secrets (modify secrets.yaml with your provider)
kubectl apply -f secrets.yaml
```

### Step 3: Apply Resource Quotas

```bash
kubectl apply -f resource-quota.yaml
```

### Step 4: Apply Network Policies

```bash
kubectl apply -f network-policy.yaml
```

### Step 5: Deploy Database

```bash
# Deploy PostgreSQL
kubectl apply -f postgres-statefulset.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n cctv-detection --timeout=300s

# Verify
kubectl get statefulset,pod,svc -n cctv-detection -l app=postgres
```

### Step 6: Deploy Redis

```bash
# Deploy Redis
kubectl apply -f redis-deployment.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app=redis -n cctv-detection --timeout=300s

# Verify
kubectl get deployment,pod,svc -n cctv-detection -l app=redis
```

### Step 7: Deploy API

```bash
# Deploy API server
kubectl apply -f api-deployment.yaml
kubectl apply -f api-service.yaml

# Wait for API to be ready
kubectl wait --for=condition=ready pod -l app=cctv-api -n cctv-detection --timeout=300s

# Verify
kubectl get deployment,pod,svc,hpa -n cctv-detection -l app=cctv-api
```

### Step 8: Deploy Ingress (Production)

```bash
# Install cert-manager (if not already installed)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager
kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s

# Deploy ingress
kubectl apply -f ingress.yaml

# Verify
kubectl get ingress,certificate -n cctv-detection
```

### Step 9: Automated Deployment Script

```bash
# Make script executable
chmod +x ../scripts/deploy-k8s.sh

# Run deployment
../scripts/deploy-k8s.sh production
```

## 🔍 Verification

### Check All Resources

```bash
# Overview
kubectl get all -n cctv-detection

# Detailed pod status
kubectl get pods -n cctv-detection -o wide

# Check events
kubectl get events -n cctv-detection --sort-by='.lastTimestamp'

# Check logs
kubectl logs -f -l app=cctv-api -n cctv-detection
```

### Test API Health

```bash
# Port forward for testing
kubectl port-forward svc/cctv-api 8000:80 -n cctv-detection

# Test health endpoint
curl http://localhost:8000/health
```

### Check Resource Usage

```bash
# Pod resource usage
kubectl top pods -n cctv-detection

# Node resource usage
kubectl top nodes

# Resource quotas
kubectl describe resourcequota -n cctv-detection
```

## 📊 Monitoring

### Prometheus Metrics

All pods expose Prometheus metrics on port 9090:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: cctv-api-metrics
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
```

### Grafana Dashboards

- **System Overview**: CPU, memory, network
- **API Metrics**: Request rate, latency, errors
- **Database Metrics**: Connections, queries, cache hit rate
- **Security Metrics**: Failed auth attempts, rate limit hits

## 🔧 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n cctv-detection

# Check events
kubectl get events -n cctv-detection

# Check logs
kubectl logs <pod-name> -n cctv-detection

# Check previous logs (if crashed)
kubectl logs <pod-name> -n cctv-detection --previous
```

### Network Issues

```bash
# Test network policies
kubectl run test-pod --image=busybox --rm -it -n cctv-detection -- sh
wget -O- http://cctv-api

# Check network policy
kubectl describe networkpolicy -n cctv-detection
```

### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it deployment/cctv-api -n cctv-detection -- sh
psql postgresql://cctv_user:password@postgres:5432/cctv_detection

# Check database logs
kubectl logs statefulset/postgres -n cctv-detection
```

### Certificate Issues

```bash
# Check certificate status
kubectl describe certificate -n cctv-detection

# Check cert-manager logs
kubectl logs -f -n cert-manager -l app=cert-manager

# Force certificate renewal
kubectl delete certificate cctv-api-cert -n cctv-detection
kubectl apply -f ingress.yaml
```

## 🎯 Production Checklist

### Before Deployment

- [ ] Secrets created and encrypted
- [ ] TLS certificates configured
- [ ] Backup strategy defined
- [ ] Monitoring configured
- [ ] Alerting rules set up
- [ ] Resource limits reviewed
- [ ] Network policies tested
- [ ] DR plan documented

### After Deployment

- [ ] Health checks passing
- [ ] Logs flowing to aggregator
- [ ] Metrics being collected
- [ ] Alerts firing correctly
- [ ] Backup jobs running
- [ ] Load testing completed
- [ ] Security scanning passed
- [ ] Documentation updated

## 🚨 Disaster Recovery

### Database Backup

```bash
# Automated backup
../scripts/backup-db.sh

# Manual backup
kubectl exec -n cctv-detection statefulset/postgres -- pg_dumpall -U postgres > backup.sql
```

### Database Restore

```bash
# Restore from backup
kubectl exec -i -n cctv-detection statefulset/postgres -- psql -U postgres < backup.sql
```

### Rollback Deployment

```bash
# Rollback to previous version
kubectl rollout undo deployment/cctv-api -n cctv-detection

# Rollback to specific revision
kubectl rollout undo deployment/cctv-api --to-revision=2 -n cctv-detection

# Check rollout history
kubectl rollout history deployment/cctv-api -n cctv-detection
```

## 📈 Scaling

### Manual Scaling

```bash
# Scale API deployment
kubectl scale deployment/cctv-api --replicas=5 -n cctv-detection

# Scale database (requires StatefulSet controller)
kubectl scale statefulset/postgres --replicas=3 -n cctv-detection
```

### Auto-scaling

Horizontal Pod Autoscaler is already configured in `api-deployment.yaml`:

```yaml
minReplicas: 3
maxReplicas: 10
targetCPUUtilizationPercentage: 70
```

## 🔐 Security Hardening

### Enable Pod Security Admission

```bash
kubectl label namespace cctv-detection \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### Enable Audit Logging

Add to API server configuration:

```yaml
--audit-log-path=/var/log/kubernetes/audit.log
--audit-policy-file=/etc/kubernetes/audit-policy.yaml
```

### Regular Security Scans

```bash
# Scan images with Trivy
trivy image ghcr.io/zyadelfeki/cctv-face-detection:latest

# Scan cluster with kubesec
kubesec scan k8s/*.yaml

# Scan with kube-bench
kube-bench run --targets master,node
```

## 📚 Additional Resources

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [RBAC Authorization](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)

## 🆘 Support

For issues or questions:
- GitHub Issues: https://github.com/zyadelfeki/cctv_face_detection/issues
- Documentation: https://github.com/zyadelfeki/cctv_face_detection/wiki
