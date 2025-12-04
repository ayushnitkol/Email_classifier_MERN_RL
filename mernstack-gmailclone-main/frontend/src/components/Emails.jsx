import React, { useEffect, useState } from 'react';
import Email from './Email';
import useGetAllEmails from '../hooks/useGetAllEmails';
import { useSelector } from 'react-redux';

const Emails = () => {
  useGetAllEmails();
  const { emails = [], searchText } = useSelector(store => store.app);
  const [filteredEmail, setFilteredEmail] = useState(emails);

  useEffect(() => {
    const search = searchText?.toLowerCase() || "";
    const filtered = emails.filter((email) => {
      return (
        (email.subject || "").toLowerCase().includes(search) ||
        (email.to || "").toLowerCase().includes(search) ||
        (email.message || "").toLowerCase().includes(search)
      );
    });
    setFilteredEmail(filtered);
  }, [searchText, emails]);

  return (
    <div className="space-y-2">
      {filteredEmail.length === 0 ? (
        <div className="py-20 text-center text-gray-400">No messages to show</div>
      ) : (
        filteredEmail.map((email) => <Email key={email._id} email={email} />)
      )}
    </div>
  );
};

export default Emails;
