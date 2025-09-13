#!/usr/bin/env python3
"""
HTTPS version of the gender detection app for phone camera access
"""
import ssl
import os
from insightface_web_app import app, initialize_insightface

def create_self_signed_cert():
    """Create self-signed certificate if it doesn't exist"""
    if not os.path.exists('cert.pem') or not os.path.exists('key.pem'):
        print("🔐 Creating self-signed SSL certificate...")
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from datetime import datetime, timedelta
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Create certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Gender Detection"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.DNSName("10.54.91.236"),
                    x509.IPAddress("127.0.0.1"),
                    x509.IPAddress("10.54.91.236"),
                ]),
                critical=False,
            ).sign(private_key, hashes.SHA256())
            
            # Write certificate and key
            with open("cert.pem", "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            with open("key.pem", "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            print("✅ SSL certificate created successfully!")
            return True
            
        except ImportError:
            print("⚠️ cryptography library not found, using OpenSSL...")
            return False
    return True

def main():
    print("🔐 Starting HTTPS Gender Detection App...")
    print("📱 Phone Access: https://10.54.91.236:5000")
    print("💻 Laptop Access: https://localhost:5000")
    print()
    
    # Initialize InsightFace
    print("🔧 Initializing InsightFace...")
    initialize_insightface()
    print()
    
    # Create SSL certificate if needed
    if not create_self_signed_cert():
        print("🔧 Using OpenSSL to create certificate...")
        os.system('openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=CA/L=SF/O=GenderDetection/CN=localhost"')
    
    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain('cert.pem', 'key.pem')
    
    print("🚀 Starting HTTPS server...")
    print("⚠️  Browser will show security warning - click 'Advanced' → 'Proceed to localhost'")
    print()
    
    # Run the app with HTTPS
    app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=False, threaded=True)

if __name__ == '__main__':
    main()
