"""TLS/HTTPS Certificate Management.

Provides:
- Self-signed certificate generation
- Let's Encrypt integration
- Automatic renewal
- Certificate validation
- Private key protection
- Certificate rotation
"""

import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import subprocess

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from loguru import logger


@dataclass
class CertificateInfo:
    """Certificate metadata."""
    subject: Dict[str, str]
    issuer: Dict[str, str]
    valid_from: datetime
    valid_until: datetime
    serial_number: int
    fingerprint: str
    is_self_signed: bool
    days_until_expiry: int
    
    def is_expired(self) -> bool:
        """Check if certificate is expired."""
        return datetime.utcnow() > self.valid_until
    
    def needs_renewal(self, days_before: int = 30) -> bool:
        """Check if certificate needs renewal."""
        return self.days_until_expiry <= days_before


class TLSCertificateManager:
    """Manages TLS certificates for HTTPS.
    
    Features:
    - Generate self-signed certificates
    - Let's Encrypt automation (via certbot)
    - Certificate validation
    - Automatic renewal
    - Private key protection
    """
    
    def __init__(
        self,
        cert_dir: str = "/etc/cctv/certs",
        renewal_days: int = 30
    ):
        """
        Initialize certificate manager.
        
        Args:
            cert_dir: Directory for storing certificates
            renewal_days: Days before expiry to trigger renewal
        """
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        
        self.renewal_days = renewal_days
        
        # Set secure permissions
        os.chmod(self.cert_dir, 0o700)
    
    def generate_private_key(self, key_size: int = 2048) -> rsa.RSAPrivateKey:
        """
        Generate RSA private key.
        
        Args:
            key_size: Key size in bits (2048 or 4096)
        
        Returns:
            RSA private key
        """
        logger.info(f"Generating {key_size}-bit RSA key")
        
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        return key
    
    def generate_self_signed_cert(
        self,
        domain: str,
        key_size: int = 2048,
        valid_days: int = 365,
        organization: str = "CCTV Face Detection",
        country: str = "US"
    ) -> Tuple[bytes, bytes]:
        """
        Generate self-signed certificate.
        
        Args:
            domain: Domain name or IP
            key_size: RSA key size
            valid_days: Certificate validity period
            organization: Organization name
            country: Country code
        
        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        logger.info(f"Generating self-signed certificate for {domain}")
        
        # Generate private key
        private_key = self.generate_private_key(key_size)
        
        # Subject and issuer (same for self-signed)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, country),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ])
        
        # Build certificate
        cert_builder = x509.CertificateBuilder()
        cert_builder = cert_builder.subject_name(subject)
        cert_builder = cert_builder.issuer_name(issuer)
        cert_builder = cert_builder.public_key(private_key.public_key())
        cert_builder = cert_builder.serial_number(x509.random_serial_number())
        cert_builder = cert_builder.not_valid_before(datetime.utcnow())
        cert_builder = cert_builder.not_valid_after(
            datetime.utcnow() + timedelta(days=valid_days)
        )
        
        # Add extensions
        cert_builder = cert_builder.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(domain),
            ]),
            critical=False
        )
        
        cert_builder = cert_builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True
        )
        
        # Sign certificate
        certificate = cert_builder.sign(private_key, hashes.SHA256())
        
        # Serialize to PEM
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        logger.info(f"Generated self-signed certificate (valid {valid_days} days)")
        return cert_pem, key_pem
    
    def save_certificate(
        self,
        cert_pem: bytes,
        key_pem: bytes,
        name: str = "default"
    ):
        """
        Save certificate and private key to disk.
        
        Args:
            cert_pem: Certificate in PEM format
            key_pem: Private key in PEM format
            name: Certificate name
        """
        cert_path = self.cert_dir / f"{name}.crt"
        key_path = self.cert_dir / f"{name}.key"
        
        # Write certificate
        with open(cert_path, 'wb') as f:
            f.write(cert_pem)
        os.chmod(cert_path, 0o644)
        
        # Write private key (secure permissions)
        with open(key_path, 'wb') as f:
            f.write(key_pem)
        os.chmod(key_path, 0o600)  # Owner read/write only
        
        logger.info(f"Saved certificate to {cert_path}")
        logger.info(f"Saved private key to {key_path} (secure permissions)")
    
    def load_certificate(self, name: str = "default") -> Optional[x509.Certificate]:
        """
        Load certificate from disk.
        
        Args:
            name: Certificate name
        
        Returns:
            Certificate object or None
        """
        cert_path = self.cert_dir / f"{name}.crt"
        
        if not cert_path.exists():
            logger.warning(f"Certificate not found: {cert_path}")
            return None
        
        with open(cert_path, 'rb') as f:
            cert_pem = f.read()
        
        certificate = x509.load_pem_x509_certificate(cert_pem)
        return certificate
    
    def get_certificate_info(self, name: str = "default") -> Optional[CertificateInfo]:
        """
        Get certificate information.
        
        Args:
            name: Certificate name
        
        Returns:
            CertificateInfo or None
        """
        cert = self.load_certificate(name)
        if not cert:
            return None
        
        # Extract subject
        subject = {
            attr.oid._name: attr.value
            for attr in cert.subject
        }
        
        # Extract issuer
        issuer = {
            attr.oid._name: attr.value
            for attr in cert.issuer
        }
        
        # Calculate days until expiry
        days_left = (cert.not_valid_after - datetime.utcnow()).days
        
        # Get fingerprint
        fingerprint = cert.fingerprint(hashes.SHA256()).hex()
        
        # Check if self-signed
        is_self_signed = cert.subject == cert.issuer
        
        return CertificateInfo(
            subject=subject,
            issuer=issuer,
            valid_from=cert.not_valid_before,
            valid_until=cert.not_valid_after,
            serial_number=cert.serial_number,
            fingerprint=fingerprint,
            is_self_signed=is_self_signed,
            days_until_expiry=days_left
        )
    
    def request_letsencrypt_cert(
        self,
        domain: str,
        email: str,
        webroot: str = "/var/www/html"
    ) -> bool:
        """
        Request Let's Encrypt certificate using certbot.
        
        Args:
            domain: Domain name
            email: Contact email
            webroot: Webroot path for ACME challenge
        
        Returns:
            True if successful
        """
        logger.info(f"Requesting Let's Encrypt certificate for {domain}")
        
        try:
            # Check if certbot is installed
            subprocess.run(['which', 'certbot'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("certbot not installed. Install with: sudo apt install certbot")
            return False
        
        try:
            # Run certbot
            cmd = [
                'certbot', 'certonly',
                '--webroot',
                '--webroot-path', webroot,
                '-d', domain,
                '--email', email,
                '--agree-tos',
                '--non-interactive'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully obtained Let's Encrypt certificate for {domain}")
                
                # Copy to our cert directory
                le_cert = f"/etc/letsencrypt/live/{domain}/fullchain.pem"
                le_key = f"/etc/letsencrypt/live/{domain}/privkey.pem"
                
                if Path(le_cert).exists() and Path(le_key).exists():
                    with open(le_cert, 'rb') as f:
                        cert_pem = f.read()
                    with open(le_key, 'rb') as f:
                        key_pem = f.read()
                    
                    self.save_certificate(cert_pem, key_pem, domain)
                
                return True
            else:
                logger.error(f"certbot failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to request Let's Encrypt certificate: {e}")
            return False
    
    def renew_certificate(self, name: str = "default") -> bool:
        """
        Renew certificate if needed.
        
        Args:
            name: Certificate name
        
        Returns:
            True if renewed or not needed
        """
        info = self.get_certificate_info(name)
        
        if not info:
            logger.error(f"Certificate {name} not found")
            return False
        
        if not info.needs_renewal(self.renewal_days):
            logger.info(
                f"Certificate {name} does not need renewal "
                f"({info.days_until_expiry} days left)"
            )
            return True
        
        logger.info(f"Certificate {name} needs renewal ({info.days_until_expiry} days left)")
        
        # If it's a Let's Encrypt cert, try to renew with certbot
        if not info.is_self_signed:
            try:
                result = subprocess.run(
                    ['certbot', 'renew'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("Successfully renewed Let's Encrypt certificate")
                    return True
                else:
                    logger.error(f"certbot renew failed: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Failed to renew with certbot: {e}")
        
        # Fallback: generate new self-signed
        logger.info("Generating new self-signed certificate")
        domain = info.subject.get('commonName', 'localhost')
        cert_pem, key_pem = self.generate_self_signed_cert(domain)
        self.save_certificate(cert_pem, key_pem, name)
        
        return True
    
    def check_all_certificates(self) -> Dict[str, CertificateInfo]:
        """
        Check all certificates in cert directory.
        
        Returns:
            Dict of certificate name -> CertificateInfo
        """
        certs = {}
        
        for cert_file in self.cert_dir.glob('*.crt'):
            name = cert_file.stem
            info = self.get_certificate_info(name)
            if info:
                certs[name] = info
                
                if info.is_expired():
                    logger.warning(f"Certificate {name} is EXPIRED!")
                elif info.needs_renewal(self.renewal_days):
                    logger.warning(
                        f"Certificate {name} needs renewal "
                        f"({info.days_until_expiry} days left)"
                    )
        
        return certs
    
    def auto_renew_all(self) -> int:
        """
        Automatically renew all certificates that need it.
        
        Returns:
            Number of certificates renewed
        """
        logger.info("Checking certificates for auto-renewal")
        
        count = 0
        certs = self.check_all_certificates()
        
        for name, info in certs.items():
            if info.needs_renewal(self.renewal_days):
                if self.renew_certificate(name):
                    count += 1
        
        logger.info(f"Auto-renewed {count} certificates")
        return count


# Global certificate manager
_cert_manager: Optional[TLSCertificateManager] = None


def get_cert_manager(
    cert_dir: str = "/etc/cctv/certs"
) -> TLSCertificateManager:
    """Get global certificate manager."""
    global _cert_manager
    
    if _cert_manager is None:
        _cert_manager = TLSCertificateManager(cert_dir)
    
    return _cert_manager
