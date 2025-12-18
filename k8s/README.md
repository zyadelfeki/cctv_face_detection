# Kubernetes Deployment - Production Ready

This directory contains production-hardened Kubernetes manifests for deploying the CCTV Face Detection system with enterprise-grade security.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Ingress Controller                       │
│                    (NGINX + cert-manager)                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
┌─────▼─────┐  ┌────▼────┐  ┌─────▼─────┐
│  API Pod  │  │ Web Pod │  │ Worker Pod│
│ (FastAPI) │  │(Streamlit)│ │(Celery)   │
└─────┬─────┘  └────┬────┘  └─────┬─────┘
      │             │              │
      └─────────────┼──────────────┘
                    │
      ┌─────────────┼─────────────┐
      │             │             │
┌─────▼─────┐ ┌────▼────┐  ┌────▼─────┐
│PostgreSQL │ │  Redis  │  │  Vault   │
│  (StatefulSet)│(StatefulSet)│(External)│
└───────────┘ └─────────┘  └──────────┘
```

## 🔒 Security Features

### 1. **Pod Security Standards**
- Enforces PSS `restricted` policy
- No privileged containers
- Read-only root filesystem
- Non-root user execution
- Dropped capabilities

### 2. **Network Policies**
- Zero-trust networking
- Explicit allow-list model
- Namespace isolation
- Ingress/egress controls

### 3. **RBAC**
- Least privilege service accounts
- Role-based access control
- No cluster-admin usage
- Audited permissions

### 4. **Secrets Management**
- External Secrets Operator integration
- HashiCorp Vault backend
- Encrypted etcd storage
- Automatic rotation support

### 5. **Resource Controls**
- CPU/memory limits and requests
- Pod disruption budgets
- Quality of Service (QoS) guarantees
- Horizontal Pod Autoscaling

## 📁 Directory Structure

```
k8s/
├── README.md                      # This file
├── namespace.yaml                 # Namespace with security labels
├── configmaps/
│   ├── app-config.yaml           # Application configuration
│   └── nginx-config.yaml         # NGINX configuration
├── secrets/
│   ├── external-secret.yaml      # External Secrets config
│   └── sealed-secrets.yaml       # Sealed Secrets (if not using Vault)
├── deployments/
│   ├── api-deployment.yaml       # FastAPI API server
│   ├── web-deployment.yaml       # Streamlit web interface
│   └── worker-deployment.yaml    # Celery worker
├── statefulsets/
│   ├── postgres-statefulset.yaml # PostgreSQL database
│   └── redis-statefulset.yaml    # Redis cache/broker
├── services/
│   ├── api-service.yaml          # API ClusterIP service
│   ├── web-service.yaml          # Web ClusterIP service
│   ├── postgres-service.yaml     # PostgreSQL headless service
│   └── redis-service.yaml        # Redis headless service
├── ingress/
│   ├── ingress.yaml              # Ingress routes
│   └── certificate.yaml          # TLS certificate
├── network-policies/
│   ├── deny-all.yaml             # Default deny policy
│   ├── api-netpol.yaml           # API network policy
│   ├── web-netpol.yaml           # Web network policy
│   ├── worker-netpol.yaml        # Worker network policy
│   ├── postgres-netpol.yaml      # PostgreSQL network policy
│   └── redis-netpol.yaml         # Redis network policy
├── rbac/
│   ├── service-accounts.yaml     # Service accounts
│   ├── roles.yaml                # Roles
│   └── role-bindings.yaml        # Role bindings
├── security/
│   ├── pod-security.yaml         # Pod Security Standards
│   ├── network-policy.yaml       # Network policies
│   └── pdb.yaml                  # Pod Disruption Budgets
├── storage/
│   ├── storage-class.yaml        # Storage class
│   └── pvc.yaml                  # Persistent Volume Claims
├── monitoring/
│   ├── servicemonitor.yaml       # Prometheus ServiceMonitor
│   └── grafana-dashboard.yaml    # Grafana dashboards
└── autoscaling/
    ├── hpa-api.yaml              # API horizontal autoscaler
    └── hpa-worker.yaml           # Worker horizontal autoscaler
```

## 🚀 Deployment Guide

### Prerequisites

1. **Kubernetes Cluster** (v1.25+)
   ```bash
   kubectl version --short
   ```

2. **Helm** (v3.0+)
   ```bash
   helm version
   ```

3. **kubectl** configured
   ```bash
   kubectl cluster-info
   ```

4. **Storage Provisioner**
   - AWS EBS CSI Driver
   - GCP Persistent Disk CSI Driver
   - Azure Disk CSI Driver
   - Or local-path-provisioner for testing

5. **Ingress Controller**
   ```bash
   # NGINX Ingress Controller
   helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
   helm install ingress-nginx ingress-nginx/ingress-nginx \
     --namespace ingress-nginx \
     --create-namespace
   ```

6. **cert-manager** (for TLS)
   ```bash
   helm repo add jetstack https://charts.jetstack.io
   helm install cert-manager jetstack/cert-manager \
     --namespace cert-manager \
     --create-namespace \
     --set installCRDs=true
   ```

7. **External Secrets Operator** (optional but recommended)
   ```bash
   helm repo add external-secrets https://charts.external-secrets.io
   helm install external-secrets external-secrets/external-secrets \
     --namespace external-secrets-system \
     --create-namespace
   ```

### Step-by-Step Deployment

#### 1. Create Namespace
```bash
kubectl apply -f namespace.yaml
```

#### 2. Configure Secrets
**Option A: Using HashiCorp Vault (Recommended)**
```bash
# Configure External Secrets to use Vault
kubectl apply -f secrets/external-secret.yaml
```

**Option B: Using Sealed Secrets**
```bash
# Install Sealed Secrets controller
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm install sealed-secrets sealed-secrets/sealed-secrets \
  --namespace kube-system

# Create sealed secret
kubectl apply -f secrets/sealed-secrets.yaml
```

#### 3. Apply ConfigMaps
```bash
kubectl apply -f configmaps/
```

#### 4. Create Storage Resources
```bash
kubectl apply -f storage/
```

#### 5. Deploy RBAC
```bash
kubectl apply -f rbac/
```

#### 6. Apply Security Policies
```bash
kubectl apply -f security/
```

#### 7. Apply Network Policies
```bash
kubectl apply -f network-policies/
```

#### 8. Deploy Stateful Services
```bash
# Deploy PostgreSQL
kubectl apply -f statefulsets/postgres-statefulset.yaml
kubectl apply -f services/postgres-service.yaml

# Deploy Redis
kubectl apply -f statefulsets/redis-statefulset.yaml
kubectl apply -f services/redis-service.yaml

# Wait for StatefulSets to be ready
kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
```

#### 9. Deploy Application
```bash
# Deploy API
kubectl apply -f deployments/api-deployment.yaml
kubectl apply -f services/api-service.yaml

# Deploy Web
kubectl apply -f deployments/web-deployment.yaml
kubectl apply -f services/web-service.yaml

# Deploy Worker
kubectl apply -f deployments/worker-deployment.yaml

# Wait for deployments
kubectl wait --for=condition=available deployment --all --timeout=300s
```

#### 10. Configure Ingress
```bash
# Create TLS certificate
kubectl apply -f ingress/certificate.yaml

# Deploy ingress
kubectl apply -f ingress/ingress.yaml
```

#### 11. Setup Autoscaling
```bash
kubectl apply -f autoscaling/
```

#### 12. Setup Monitoring (Optional)
```bash
kubectl apply -f monitoring/
```

### Verification

```bash
# Check all resources
kubectl get all -n cctv-face-detection

# Check pod status
kubectl get pods -n cctv-face-detection

# Check services
kubectl get svc -n cctv-face-detection

# Check ingress
kubectl get ingress -n cctv-face-detection

# Check network policies
kubectl get networkpolicies -n cctv-face-detection

# View logs
kubectl logs -f deployment/api -n cctv-face-detection
kubectl logs -f deployment/web -n cctv-face-detection
kubectl logs -f deployment/worker -n cctv-face-detection
```

## 🔧 Configuration

### Environment Variables

Set these in `configmaps/app-config.yaml`:

```yaml
DATABASE_URL: "postgresql://user:password@postgres:5432/cctv_db"
REDIS_URL: "redis://redis:6379/0"
LOG_LEVEL: "INFO"
ENVIRONMENT: "production"
```

### Resource Limits

Adjust in deployment manifests:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Autoscaling Thresholds

Modify in `autoscaling/hpa-*.yaml`:

```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
```

## 🔄 Updates and Rollbacks

### Rolling Update
```bash
# Update image
kubectl set image deployment/api \
  api=your-registry/cctv-api:v2.0.0 \
  -n cctv-face-detection

# Check rollout status
kubectl rollout status deployment/api -n cctv-face-detection
```

### Rollback
```bash
# View rollout history
kubectl rollout history deployment/api -n cctv-face-detection

# Rollback to previous version
kubectl rollout undo deployment/api -n cctv-face-detection

# Rollback to specific revision
kubectl rollout undo deployment/api --to-revision=2 -n cctv-face-detection
```

## 📊 Monitoring

### Metrics
- Prometheus scrapes `/metrics` endpoint
- Grafana dashboards in `monitoring/`
- ServiceMonitor CRDs for automatic discovery

### Health Checks
- Liveness probe: `/health/live`
- Readiness probe: `/health/ready`
- Startup probe: `/health/startup`

### Logs
```bash
# Stream logs
kubectl logs -f deployment/api -n cctv-face-detection

# View logs from all pods
kubectl logs -l app=api -n cctv-face-detection --tail=100

# Export logs
kubectl logs deployment/api -n cctv-face-detection > api.log
```

## 🔐 Security Best Practices

1. **Always use Pod Security Standards**
   - Enforce `restricted` policy in production
   - Block privileged containers
   - Require read-only root filesystem

2. **Implement Network Policies**
   - Start with deny-all
   - Add explicit allow rules
   - Segment by namespace

3. **Use Secrets Management**
   - Never commit secrets to Git
   - Use External Secrets Operator
   - Rotate secrets regularly

4. **Resource Limits**
   - Set both requests and limits
   - Prevent resource exhaustion
   - Use Pod Disruption Budgets

5. **Regular Updates**
   - Keep K8s version current
   - Update container images
   - Scan for vulnerabilities

## 🆘 Troubleshooting

### Pod Not Starting
```bash
kubectl describe pod <pod-name> -n cctv-face-detection
kubectl logs <pod-name> -n cctv-face-detection --previous
```

### Network Issues
```bash
# Test connectivity
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- /bin/bash

# Check network policy
kubectl describe networkpolicy -n cctv-face-detection
```

### Storage Issues
```bash
kubectl get pv
kubectl get pvc -n cctv-face-detection
kubectl describe pvc <pvc-name> -n cctv-face-detection
```

### Performance Issues
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n cctv-face-detection

# Check HPA status
kubectl get hpa -n cctv-face-detection
kubectl describe hpa api -n cctv-face-detection
```

## 📚 Additional Resources

- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [External Secrets Operator](https://external-secrets.io/)

## 🔄 Backup and Disaster Recovery

### Database Backup
```bash
# Create backup job
kubectl apply -f backup/postgres-backup-cronjob.yaml

# Manual backup
kubectl create job --from=cronjob/postgres-backup manual-backup-$(date +%s)
```

### Restore from Backup
```bash
# See backup/restore-procedure.md
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Manual scaling
kubectl scale deployment api --replicas=5 -n cctv-face-detection

# Auto-scaling is configured via HPA
```

### Vertical Scaling
```bash
# Update resource limits in deployment manifests
kubectl apply -f deployments/api-deployment.yaml
```

---

**Maintained by**: Security Team  
**Last Updated**: 2025-12-18  
**Version**: 1.0.0
